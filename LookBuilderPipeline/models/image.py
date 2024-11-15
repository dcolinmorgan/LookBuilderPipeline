from sqlalchemy import Column, Integer, DateTime, ForeignKey, Float, Boolean, String
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

        # Add this line with your other column definitions:
    image_type = Column(String(10), nullable=False)  # Add this line
    updated_at = Column(DateTime)  # Add this
    processed = Column(Boolean, default=False)  # Add this

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

    def get_resize_variant(self, size: int, aspect_ratio: float = 1.0, square: bool = False):
        """Get an existing resize variant with the specified parameters."""
        # Import here to avoid circular imports
        from .image_variant import ImageVariant
        
        return (
            self.variants
            .filter(
                ImageVariant.parameters['size'].astext.cast(Integer) == size,
                ImageVariant.parameters['aspect_ratio'].astext.cast(Float) == aspect_ratio,
                ImageVariant.parameters['square'].astext.cast(Boolean) == square
            )
            .first()
        )

    def get_or_create_resize_variant(self, session, size: int, aspect_ratio: float = 1.0, square: bool = False):
        """Get an existing resize variant or create a new one if it doesn't exist."""
        # Import here to avoid circular imports
        from .image_variant import ImageVariant
        
        # First try to get an existing variant
        variant = self.get_resize_variant(size, aspect_ratio, square)
        if variant:
            return variant
            
        # If no variant exists, create a new one
        image_data = self.get_image_data(session)
        if not image_data:
            raise ValueError(f"Image {self.image_id} has no data")
            
        # Import the resize_images function using relative import
        from ..utils.resize import resize_images
        
        # Create the resized image
        resized_image = resize_images(image_data, size, aspect_ratio, square)
        
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
        variant = ImageVariant(
            source_image_id=self.image_id,
            variant_oid=oid,
            variant_type='resized',
            parameters={
                'size': size,
                'aspect_ratio': aspect_ratio,
                'square': square
            },
            processed=True
        )
        
        # Add and flush to get the variant_id
        session.add(variant)
        session.flush()
        
        return variant

    def __repr__(self):
        return f"<Image(image_id={self.image_id}, oid={self.image_oid})>"