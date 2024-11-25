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
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from psycopg2.extensions import register_adapter, AsIs
from .image_relationships import look_images

class VariantHandler(ABC):
    """Base class for handling different variant types."""
    
    def __init__(self, image, session):
        self.image = image
        self.session = session
        
    @abstractmethod
    def get_filter_conditions(self, params: Dict[str, Any]) -> List:
        """Get filter conditions for finding existing variants."""
        pass
        
    @abstractmethod
    def process_image(self, image_data: bytes, params: Dict[str, Any]) -> bytes:
        """Process the image data."""
        pass
        
    @abstractmethod
    def get_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters to store with variant."""
        pass

class ResizeHandler(VariantHandler):
    def get_filter_conditions(self, params):
        return [
            ImageVariant.variant_type == 'resize',
            ImageVariant.source_image_id == self.image.image_id,
            ImageVariant.parameters['size'].astext.cast(Integer) == params['size'],
            ImageVariant.parameters['aspect_ratio'].astext.cast(Float) == params.get('aspect_ratio', 1.0),
            ImageVariant.parameters['square'].astext.cast(Boolean) == params.get('square', False)
        ]
        
    def process_image(self, image_data, params):
        from LookBuilderPipeline.utils.resize import resize_images
        return resize_images(
            image_data,
            self.get_parameters(params)
        )
        
    def get_parameters(self, params):
        return {
            'size': params['size'],
            'aspect_ratio': params.get('aspect_ratio', 1.0),
            'square': params.get('square', False)
        }

class PoseHandler(VariantHandler):
    def get_filter_conditions(self, params):
        return [
            ImageVariant.variant_type == 'pose',
            ImageVariant.source_image_id == self.image.image_id,
            ImageVariant.parameters['face'].astext.cast(Boolean) == params.get('face', True)
        ]
        
    def process_image(self, image_data, params):
        from LookBuilderPipeline.pose import detect_pose
        return detect_pose(self=None,
            image_path=image_data,
            **self.get_parameters(params)
        )
        
    def get_parameters(self, params):
        return {'face': params.get('face', True)}

class SegmentHandler(VariantHandler):
    def get_filter_conditions(self, params):
        return [
            ImageVariant.variant_type == 'segment',
            ImageVariant.source_image_id == self.image.image_id,
            ImageVariant.parameters['inverse'].astext.cast(Boolean) == params.get('inverse', True)
        ]
        
    def process_image(self, image_data, params):
        from LookBuilderPipeline.segment import segment_image
        return segment_image(self=None,
            image_path=image_data, 
            **self.get_parameters(params)
        )
        
    def get_parameters(self, params):
        return {'inverse': params.get('inverse', True)}

class SDXLHandler(VariantHandler):
    def get_filter_conditions(self, params):
        return [
            ImageVariant.variant_type == 'sdxl',
            ImageVariant.source_image_id == self.image.image_id,
            ImageVariant.parameters['prompt'].astext.cast(String) == params.get('prompt', ''),
            ImageVariant.parameters['neg_prompt'].astext.cast(String) == params.get('negative_prompt', 'ugly, deformed')
        ]
        
    def process_image(self, image_data, params):
        from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
        model = ImageModelSDXL(
            image=image_data, 
            **self.get_parameters(params)
        )
        model.prepare_model()
        return model.generate_image()
        
    def get_parameters(self, params):
        return {
            'pose': params['image_pose_id'],
            'mask': params['image_segment_id'],
            'prompt': params.get('prompt', ''),
            'neg_prompt': params.get('negative_prompt', 'ugly, deformed')
        }

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

    @classmethod
    def get_db_manager(cls):
        from LookBuilderPipeline.manager.db_manager import DBManager
        return DBManager()

    @classmethod
    def get_by_id(cls, image_id: int):
        """Get an image by its ID"""
        db_manager = cls.get_db_manager()
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

    def _get_handler(self, variant_type: str, session) -> VariantHandler:
        """Get the appropriate handler for a variant type.
        
        Args:
            variant_type: Type of variant to handle
            session: Database session
            
        Returns:
            VariantHandler: Handler instance for the variant type
        """
        handlers = {
            'resize': ResizeHandler,
            'pose': PoseHandler,
            'segment': SegmentHandler,
            'sdxl': SDXLHandler
        }
        
        handler_class = handlers.get(variant_type)
        if not handler_class:
            raise ValueError(f"Unsupported variant type: {variant_type}")
            
        return handler_class(self, session)

    def get_variant(self, variant_type: str, session, **kwargs) -> Optional['ImageVariant']:
        """Get an existing variant if it exists."""
        handler = self._get_handler(variant_type, session)
        filter_conditions = handler.get_filter_conditions(kwargs)
        
        return self.variants.filter(*filter_conditions).first()

    def get_variant_image(self, variant: 'ImageVariant', session) -> Optional[bytes]:
        """Get the actual image data from a variant.
        
        Args:
            variant: ImageVariant instance to get image from
            session: Database session
            
        Returns:
            bytes: Image data if found, None otherwise
        """
        if not variant:
            logging.error("No variant provided to get_variant_image")
            return None
            
        try:
            logging.info(f"Getting image data for variant {variant.id}")
            connection = session.connection().connection
            
            lob = connection.lobject(oid=variant.variant_oid, mode='rb')
            data = lob.read()
            lob.close()
            
            logging.info(f"Successfully read {len(data)} bytes from variant {variant.id}")
            return data
            
        except Exception as e:
            logging.error(f"Error reading variant image data: {str(e)}", exc_info=True)
            return None

    def create_variant(self, variant_type: str, session, **kwargs) -> 'ImageVariant':
        """Create a new variant of the specified type."""
        handler = self._get_handler(variant_type, session)
        
        # Get image data
        image_data = self.get_image_data(session)
        if not image_data:
            raise ValueError(f"Image {self.image_id} has no data")
            
        # Process the image
        processed_image = handler.process_image(image_data, kwargs)
        
        # Convert to bytes if needed
        if not isinstance(processed_image, bytes):
            from io import BytesIO
            img_byte_arr = BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            processed_image = img_byte_arr.getvalue()
        
        # Create variant
        parameters = {key: value for key, value in kwargs.items() if key not in ['pose', 'segment']}
        variant = ImageVariant(
            source_image_id=self.image_id,
            parameters=parameters,
            variant_type=variant_type,
            processed=True
        )
        
        session.add(variant)
        session.flush()
        
        # Store as large object
        lob = session.connection().connection.lobject(mode='wb')
        lob.write(processed_image)
        variant.variant_oid = lob.oid
        lob.close()
        
        return variant

    def get_or_create_variant(self, variant_type: str, session, **kwargs) -> 'ImageVariant':
        """Get an existing variant or create a new one."""
        variant = self.get_variant(variant_type, session, **kwargs)
        if variant:
            return variant
            
        return self.create_variant(variant_type, session, **kwargs)

    # Convenience methods
    def get_or_create_resize_variant(self, session, size: int, aspect_ratio: float = 1.0, square: bool = False):
        return self.get_or_create_variant('resize', session, size=size, aspect_ratio=aspect_ratio, square=square)

    def get_or_create_pose_variant(self, session, face: bool = True):
        return self.get_or_create_variant('pose', session, face=face)

    def get_or_create_segment_variant(self, session, inverse: bool = False):
        return self.get_or_create_variant('segment', session, inverse=inverse)

    def get_or_create_sdxl_variant(self, session, prompt: str, negative_prompt: str = 'ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves'):
        segment_variant = self.get_variant('segment', session)
        segment_variant = self.get_variant_image(segment_variant, session)
        
        pose_variant = self.get_variant('pose', session)
        pose_variant = self.get_variant_image(pose_variant, session)
        
        return self.get_or_create_variant('sdxl', session, image_segment_id=segment_variant, image_pose_id=pose_variant, prompt=prompt, neg_prompt=negative_prompt)

    @classmethod
    def get_unassigned_images(cls, user_id: int = None):
        """Get all images that aren't associated with any look"""
        db_manager = cls.get_db_manager()
        with db_manager.get_session() as session:
            query = session.query(cls).outerjoin(look_images).filter(look_images.c.look_id == None)
            
            # If user_id is provided, filter by user
            if user_id is not None:
                query = query.filter(cls.user_id == user_id)
                
            images = query.all()
            for image in images:
                session.expunge(image)
            return images

    def __repr__(self):
        return f"<Image(image_id={self.image_id}, oid={self.image_oid})>"
