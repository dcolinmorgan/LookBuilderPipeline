from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Boolean, cast, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import Float, Boolean
from datetime import datetime
from .base import Base
from .image_variant import ImageVariant
from LookBuilderPipeline.models.process_queue import ProcessQueue
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql import func
from typing import Optional
import logging
import numpy as np
import os
import cv2

class Image(Base):
    __tablename__ = 'images'

    image_id = Column(Integer, primary_key=True)
    image_oid = Column(Integer)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.now)

    image_type = Column(String(10), nullable=False)  
    updated_at = Column(DateTime) 
    processed = Column(Boolean, default=False)  

    # Use string reference for User
    user = relationship("User", back_populates="images")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_db_manager(self):
        from LookBuilderPipeline.manager.db_manager import DBManager
        return DBManager()

    @classmethod
    def get_by_id(cls, image_id: int):
        """Get an image by its ID"""
        db_manager = DBManager()
        with db_manager.get_session() as session:
            image = session.query(cls).get(image_id)
            if image:
                session.expunge(image)
            return image

    def save(self):
        """Save the image to the database"""
        db_manager = self.get_db_manager()
        with db_manager.get_session() as session:
            session.add(self)
            session.flush()
            image_id = self.image_id
            session.expunge(self)
            return image_id

    def update(self, **kwargs):
        """Update image attributes"""
        db_manager = self.get_db_manager()
        with db_manager.get_session() as session:
            session.add(self)
            for key, value in kwargs.items():
                setattr(self, key, value)
            session.flush()
            session.expunge(self)

    def get_image_data(self, session):
        """Get the image data from the large object storage."""
        logging.info(f"Attempting to get image data for image_id={self.image_id}, image_oid={self.image_oid}")
        
        if not self.image_oid:
            logging.error(f"No image_oid found for image {self.image_id}")
            return None
            
        try:
            logging.info(f"Creating lobject for image {self.image_id} with oid {self.image_oid}")
            connection = session.connection().connection
            
            lob = connection.lobject(oid=self.image_oid, mode='rb')
            logging.info(f"Successfully created lobject for image {self.image_id}")
            
            data = lob.read()
            logging.info(f"Successfully read {len(data)} bytes from image {self.image_id}")
            
            lob.close()
            return data
            
        except Exception as e:
            logging.error(f"Error reading image data for image {self.image_id}: {str(e)}", exc_info=True)
            return None

    def get_variant(self, session, variant_type: str, **kwargs):
        """Get an existing variant or create a new one if it doesn't exist."""
        # Import here to avoid circular imports
        from .image_variant import ImageVariant
        
        # Determine the filter parameters based on variant type
        if variant_type == 'resize':
            size = kwargs.get('size')
            aspect_ratio = kwargs.get('aspect_ratio', 1.0)
            square = kwargs.get('square', False)
            filter_conditions = [
                ImageVariant.parameters['size'].astext.cast(Integer) == size,
                ImageVariant.parameters['aspect_ratio'].astext.cast(Float) == aspect_ratio,
                ImageVariant.parameters['square'].astext.cast(Boolean) == square
            ]
        elif variant_type == 'pose':
            face = kwargs.get('face', True)
            filter_conditions = [
                ImageVariant.variant_type == 'pose',
                ImageVariant.source_image_id == self.image_id,
                ImageVariant.parameters['face'].astext.cast(Boolean) == face
            ]
        elif variant_type == 'segment':
            inverse = kwargs.get('inverse', True)
            filter_conditions = [
                ImageVariant.parameters['inverse'].astext.cast(Boolean) == inverse,
            ]
        elif variant_type == 'loadpipe':
            pipe = kwargs.get('pipe')
            filter_conditions = [
                ImageVariant.parameters['pipe'].astext.cast(String) == pipe,
            ]
        elif variant_type == 'runpipe':
            pipe = kwargs.get('pipe')
            prompt = kwargs.get('prompt')
            filter_conditions = [
                ImageVariant.parameters['pipe'].astext.cast(String) == pipe,
                ImageVariant.parameters['prompt'].astext.cast(String) == prompt,
            ]
        else:
            raise ValueError("Unsupported variant type")

        # First try to get an existing variant
        variant = self.variants.filter(*filter_conditions).first()
        if variant:
            return variant
            
        # If no variant exists, create a new one
        image_data = self.get_image_data(session)
        if not image_data:
            raise ValueError(f"Image {self.image_id} has no data")
        
        # Fix the process query - check if process_id exists first
        process_id = kwargs.get('process_id')
        process = None
        if process_id is not None:
            process = session.query(ProcessQueue).filter(ProcessQueue.process_id == process_id).first()

        # Import the appropriate function based on variant type
        if variant_type == 'resize':
            from LookBuilderPipeline.resize import resize_images
            resized_image = resize_images(image_data, kwargs['size'], aspect_ratio, square)
        elif variant_type == 'pose':
            from dwpose import DwposeDetector
            import tempfile
            from PIL import Image
            import io
            import torch
            import logging
            import numpy as np
            import os

            try:
                logging.info(f"Starting pose detection for image {self.image_id}")
                
                # Convert bytes to numpy array using cv2
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                logging.info(f"Loaded source image with shape: {image.shape}")
                
                logging.info(f"Loading DWPose model...")
                model = DwposeDetector.from_pretrained_default()
                logging.info(f"Model loaded successfully")
                
                logging.info(f"Running pose detection...")
                # Pass the numpy array directly to the model
                pose_image, json_data, source = model(image,
                    include_hand=True,
                    include_face=kwargs.get('face', True),
                    include_body=True,
                    image_and_json=True)
                
                logging.info(f"Pose detection completed. Pose image type: {type(pose_image)}")
                
                # Save the pose image to a debug file
                debug_output_dir = 'debug_poses'
                os.makedirs(debug_output_dir, exist_ok=True)
                debug_path = os.path.join(debug_output_dir, f'pose_debug_{self.image_id}.png')
                
                # Convert numpy array to PIL Image if necessary
                if isinstance(pose_image, np.ndarray):
                    logging.info("Converting numpy array to PIL Image")
                    # Convert BGR to RGB if necessary
                    if pose_image.shape[2] == 3:
                        pose_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
                
                # Save debug image
                pose_image.save(debug_path)
                logging.info(f"Saved debug pose image to: {debug_path}")
                
                # Convert to bytes for database storage
                img_byte_arr = io.BytesIO()
                pose_image.save(img_byte_arr, format='PNG')
                pose_image_bytes = img_byte_arr.getvalue()
                
                # Create and store variant
                variant = ImageVariant(
                    source_image_id=self.image_id,
                    variant_type='pose',
                    parameters={
                        'face': kwargs.get('face', True),
                        'pose_data': json_data  # Store the pose data for later use
                    },
                    processed=True
                )
                session.add(variant)
                session.flush()
                
                # Store the image using lobject
                lob = session.connection().connection.lobject(mode='wb')
                lob.write(pose_image_bytes)
                variant.variant_oid = lob.oid
                lob.close()
                
                logging.info(f"Successfully created pose variant for image {self.image_id}")
                return variant
                
            except Exception as e:
                logging.error(f"Error in pose detection: {str(e)}", exc_info=True)
                raise
            finally:
                if 'temp_path' in locals():
                    import os
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logging.error(f"Error cleaning up temp file: {str(e)}")
        elif variant_type == 'segment':
            from LookBuilderPipeline.segment import segment_image
            resized_image = segment_image(image_data, inverse=kwargs.get('inverse', False))
        elif variant_type == 'loadpipe':
            if process.parameters.get('pipe') == 'sdxl':
                from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
                model = ImageModelSDXL()  # Create instance
                model.prepare_model()
                model.prepare_image({
                    'image_path': kwargs.get('image_path'),
                    'pose_path': kwargs.get('pose_path'),
                    'mask_path': kwargs.get('mask_path')
                })
            elif process.parameters.get('pipe') == 'flux':
                from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux
                model = ImageModelFlux()  # Create instance
                model.prepare_model()
                model.prepare_image({
                    'image_path': kwargs.get('image_path'),
                    'pose_path': kwargs.get('pose_path'),
                    'mask_path': kwargs.get('mask_path')
                })
        elif variant_type == 'runpipe':
            if process.parameters.get('pipe') == 'sdxl':
                from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
                self.ImageModelSDXL.generate_image()
            elif process.parameters.get('pipe') == 'flux':
                from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux
                self.ImageModelFlux.generate_image()
               

        
        # Convert the resized image to bytes if it isn't already
        if not isinstance(resized_image, bytes):
            from io import BytesIO
            img_byte_arr = BytesIO()
            resized_image.save(img_byte_arr, format='PNG')  # or 'JPEG' depending on your needs
            resized_image = img_byte_arr.getvalue()
        
        # Store the resized image as a large object
        lob = session.connection().connection.lobject(mode='wb')
        lob.write(resized_image)
        oid = lob.oid  # Get the OID before closing
        lob.close()
        
        # Create the variant with parameters in JSONB
        parameters = kwargs if variant_type == 'resize' else {'face': kwargs['face']}
        variant = ImageVariant(
            source_image_id=self.image_id,
            variant_oid=oid,
            variant_type=variant_type,
            parameters=parameters,
            processed=True
        )
        
        # Add and flush to get the variant_id
        session.add(variant)
        session.flush()
        
        return variant

    def get_or_create_resize_variant(self, session, size: int, aspect_ratio: float = 1.0, square: bool = False):
        """Get or create a resize variant."""
        return self.get_variant(session, 'resize', size=size, aspect_ratio=aspect_ratio, square=square)

    def get_or_create_pose_variant(self, session, face=True):
        """Get or create a pose variant for the image."""
        return self.get_variant(session, 'pose', face=face)

    def get_or_create_segment_variant(self, session, inverse: bool):
        """Get or create a segment variant."""
        return self.get_variant(session, 'segment', inverse=inverse)
    
    def load_pipe_variant(self, session, model: str):
        """Load pipe variant."""
        return self.get_variant(session, 'loadpipe', model=model)
    
    def run_pipe_variant(self, session, model: str, prompt: str):
        """Runpipe variant."""
        return self.get_variant(session, 'runpipe', model=model, prompt=prompt)

    def __repr__(self):
        return f"<Image(image_id={self.image_id}, oid={self.image_oid})>"
