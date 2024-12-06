from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import OID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from PIL import Image as PILImage
import io
from typing import Optional
import logging
from datetime import datetime
from .base import Base
from .image_variant import ImageVariant
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
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

    def get_or_create_variant(self, session, variant_type: str, **kwargs):
        """Get an existing variant or create a new one if it doesn't exist."""
        from .image_variant import ImageVariant  # Import here to avoid circular import
        
        # Look for existing variant with matching parameters
        existing_variant = (
            session.query(ImageVariant)
            .filter_by(
                source_image_id=self.image_id,
                variant_type=variant_type,
                parameters=kwargs
            )
            .first()
        )

        if existing_variant:
            return existing_variant

        # Create new variant if none exists
        logging.info(f"Creating new variant of type {variant_type}")
        variant = self.create_variant(
            variant_type,
            session,
            **kwargs
        )
        logging.info(f"Variant created of type {variant_type}")

        # # Convert PIL Image back to bytes
        # output = io.BytesIO()
        # variant.save(output, format='PNG')
        # variant_bytes = output.getvalue()

        # # Store the variant image
        # variant_oid = self.store_large_object(session, variant_bytes)

        # # Create and return the new variant
        # variant = ImageVariant(
        #     source_image_id=self.image_id,
        #     variant_oid=variant_oid,
        #     variant_type=variant_type,
        #     parameters=kwargs,
        #     user_id=self.user_id
        # )
        # session.add(variant)
        # session.flush()  # Ensure variant_id is generated
        return variant

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


    def get_variant(self, variant_type: str, session, **kwargs):
        """Get a variant with specific parameters"""
        # Merge the image with the current session
        session.add(self)
        
        filter_conditions = [
            ImageVariant.source_image_id == self.image_id,
            ImageVariant.variant_type == variant_type
        ]
        
        for key, value in kwargs.items():
            if key == 'size':
                # Handle size as single integer
                filter_conditions.append(
                    cast(ImageVariant.parameters['size'].astext, Integer) == value
                )
            else:
                # Handle other parameters normally
                if isinstance(value, bool):
                    filter_conditions.append(
                        cast(ImageVariant.parameters[key].astext, Boolean) == value
                    )
                elif isinstance(value, float):
                    filter_conditions.append(
                        cast(ImageVariant.parameters[key].astext, Float) == value
                    )
                else:
                    filter_conditions.append(
                        cast(ImageVariant.parameters[key].astext, String) == str(value)
                    )
        
        try:
            return self.variants.filter(*filter_conditions).first()
        finally:
            # Clean up by expunging the image from session
            session.expunge(self)


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
        logging.info(f"Creating variant of type {variant_type}")
        image_data = self.get_image_data(session)
        logging.info(f"kwargs: {kwargs}")
        
        image_model = BaseImageModel(image= image_data, **kwargs)
        logging.info(f"Image model device: {image_model.device}")
        
        if variant_type == 'segment':
            processed_image = image_model.mask
        elif variant_type == 'pose':
            processed_image = image_model.pose
        elif variant_type == 'outfit':
            processed_image = image_model.outfit
        elif variant_type == 'sdxl':
            sdxl_model = ImageModelSDXL(image_model.image, **kwargs) 
            sdxl_model.prepare_model()
            processed_image = sdxl_model.generate_image()
        elif variant_type == 'flux':
            flux_model = ImageModelFlux(image_model.image, **kwargs)
            flux_model.prepare_model()
            processed_image = flux_model.generate_image()
        else:
            raise ValueError(f"Invalid variant type: {variant_type}")
        
        # Get image data
        # image_data = self.get_image_data(session)
        # if not image_data:
            # raise ValueError(f"Image {self.image_id} has no data")
            
        # Process the image

        # processed_image = handler.process_image(image_data, kwargs)
        # Convert to bytes if needed
        if not isinstance(processed_image, bytes):
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            processed_image = img_byte_arr.getvalue()
        
        # Create variant
        parameters = {key: value for key, value in kwargs.items() if key not in ['image_pose_id','image_segment_id','pose', 'segment']}
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
