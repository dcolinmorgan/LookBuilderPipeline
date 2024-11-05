from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import OID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from PIL import Image as PILImage
from io import BytesIO

from .base import Base
from ..utils.resize import resize_images

class Image(Base):
    __tablename__ = 'images'

    image_id = Column(Integer, primary_key=True)
    image_oid = Column(OID, nullable=False)
    image_type = Column(String(10), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    processed = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)

    def get_or_create_resize_variant(self, session, size: int, aspect_ratio: float = 1.0, square: bool = False):
        """Get an existing resize variant or create a new one if it doesn't exist."""
        from .image_variant import ImageVariant  # Import here to avoid circular import
        
        # Look for existing variant with matching parameters
        existing_variant = (
            session.query(ImageVariant)
            .filter_by(
                source_image_id=self.image_id,
                variant_type='resized',
                parameters={
                    'size': size,
                    'aspect_ratio': aspect_ratio,
                    'square': square
                }
            )
            .first()
        )

        if existing_variant:
            return existing_variant

        # Create new variant if none exists
        image_bytes = self.get_image_data(session)
        resized_image = resize_images(
            image_bytes,
            target_size=size,
            aspect_ratio=aspect_ratio,
            square=square
        )

        # Convert resized PIL Image back to bytes
        output = BytesIO()
        resized_image.save(output, format='PNG')
        resized_bytes = output.getvalue()

        # Store the resized image
        variant_oid = self.store_large_object(session, resized_bytes)

        # Create and return the new variant
        variant = ImageVariant(
            source_image_id=self.image_id,
            variant_oid=variant_oid,
            variant_type='resized',
            parameters={
                'size': size,
                'aspect_ratio': aspect_ratio,
                'square': square
            },
            user_id=self.user_id
        )
        session.add(variant)
        session.flush()  # Ensure variant_id is generated
        return variant

    def get_image_data(self, session) -> bytes:
        """Get the image data from the database."""
        # Implementation depends on your database setup
        # This should retrieve the image data using self.image_oid
        pass

    def store_large_object(self, session, data: bytes) -> int:
        """Store image data as a large object and return its OID."""
        # Implementation depends on your database setup
        pass