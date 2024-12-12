from typing import Optional, List, Union, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    # Import Image only for type hints to avoid circular imports
    from .image import Image

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base

class ImageVariant(Base):
    __tablename__ = 'image_variants'
    
    variant_id = Column(Integer, primary_key=True)
    source_image_id = Column(Integer, ForeignKey('images.image_id'), nullable=False)
    parent_variant_id = Column(Integer, ForeignKey('image_variants.variant_id'), nullable=True)
    variant_oid = Column(Integer, nullable=True)
    variant_type = Column(String, nullable=False)
    parameters = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    processed = Column(Boolean, default=False)
   class_type  = Column(String, nullable=False, default='image_variant')

    __table_args__ = (
        Index('idx_image_variants_source_params', 
              'source_image_id', 
              text('(parameters->\'face\')::boolean')),
    )

    __mapper_args__ = {
        'polymorphic_on': class_type,
        'polymorphic_identity': 'image_variant'
    }

    # Relationships
    source_image = relationship("Image", back_populates="variants")
    
    @property
    def user(self):
        """Get the user through the source image."""
        return self.source_image.user if self.source_image else None

    def __repr__(self):
        return f"<ImageVariant(variant_id={self.variant_id}, type={self.variant_type})>"
 
    def get_variant_chain(self) -> List[Union["ImageVariant", "Image"]]:
        """Get the full chain of variants leading to this image."""
        chain = []
        current = self
        while current.parent_variant:
            chain.append(current.parent_variant)
            current = current.parent_variant
        chain.append(current.source_image)
        return list(reversed(chain))
    
    @property
    def size(self) -> Optional[int]:
        """Get the size parameter for resized variants."""
        if self.variant_type == 'resized' and self.parameters:
            return self.parameters.get('size')
        return None

    @classmethod
    def find_by_size(cls, session, source_image_id: int, size: int) -> Optional["ImageVariant"]:
        """Find a resized variant of specific size for an image."""
        return (session.query(cls)
                .filter(cls.source_image_id == source_image_id,
                       cls.variant_type == 'resized',
                       cls.parameters['size'].astext.cast(Integer) == size)
                .first())
    @classmethod
    def get_all_variants(cls, session, source_image_id: int, variant_type: Optional[str] = None) -> List["ImageVariant"]:
        """Get all variants for an image, optionally filtered by type."""
        query = session.query(cls).filter(cls.source_image_id == source_image_id)
        if variant_type:
            query = query.filter(cls.variant_type == variant_type)
        return query.all() 

    @property
    def id(self):
        """Alias for variant_id for compatibility."""
        return self.variant_id
