from sqlalchemy import Column, Integer, DateTime, ForeignKey, Float, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import Float, Boolean
from datetime import datetime
from .base import Base
import logging

class Image(Base):
    __tablename__ = 'images'

    image_id = Column(Integer, primary_key=True)
    image_oid = Column(Integer)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    created_at = Column(DateTime, default=datetime.now)

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
            face = kwargs.get('face',True)
            filter_conditions = [
                ImageVariant.parameters['face'].astext.cast(Integer) == face,
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
        process = session.query(ProcessQueue).get(data['process_id'])
        # Import the appropriate function based on variant type
        if variant_type == 'resize':
            from LookBuilderPipeline.resize import resize_images
            resized_image = resize_images(image_data, kwargs['size'], aspect_ratio, square)
        elif variant_type == 'pose':
            from LookBuilderPipeline.pose import detect_pose
            resized_image = detect_pose(image_data)
        elif variant_type == 'segment':
            from LookBuilderPipeline.segment import segment_image
            resized_image = segment_image(image_data, inverse=kwargs.get('inverse', False))
        elif variant_type == 'loadpipe':
            if process.parameters.get('pipe') == 'sdxl':
                from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
                self.ImageModelSDXL.prepare_model()
                self.ImageModelSDXL.prepare_image(full_data.keys(['image_path','pose_path','mask_path']))
            elif process.parameters.get('pipe') == 'flux':
                from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux
                self.ImageModelFlux.prepare_model()
                self.ImageModelFlux.prepare_image(full_data.keys(['image_path','pose_path','mask_path']))
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

    def get_or_create_pose_variant(self, session, face: bool):
        """Get or create a pose variant."""
        return self.get_variant(session, 'pose', face=face)
    
    def get_or_create_segment_variant(self, session, inverse: bool):
        """Get or create a segment variant."""
        return self.get_variant(session, 'segment', inverse=inverse)
    
    def load_pipe_variant(self, session, model: str):
        """Load pipe variant."""
        return self.get_variant(session, 'loadpipe', model=model)
    
    def run_pipe_variant(self, session, model: str):
        """Runpipe variant."""
        return self.get_variant(session, 'runpipe', model=model, prompt=prompt)

    def __repr__(self):
        return f"<Image(image_id={self.image_id}, oid={self.image_oid})>"
