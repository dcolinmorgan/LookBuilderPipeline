from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base

class ImageVariant(Base):
    __tablename__ = 'image_variants'
    
    variant_id = Column(Integer, primary_key=True)
    source_image_id = Column(Integer, ForeignKey('images.image_id'), nullable=False)
    parent_variant_id = Column(Integer, ForeignKey('image_variants.variant_id'))
    variant_oid = Column(Integer, nullable=False)
    variant_type = Column(String(20), nullable=False)
    parameters = Column(JSONB, default={})
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    processed = Column(Boolean, default=False)

    # Relationships
    source_image = relationship("Image", back_populates="variants")
    
    @property
    def user(self):
        """Get the user through the source image."""
        return self.source_image.user if self.source_image else None

    def __repr__(self):
        return f"<ImageVariant(variant_id={self.variant_id}, type={self.variant_type})>"
 
    def get_variant_chain(self):
        """Get the full chain of variants leading to this image."""
        chain = []
        current = self
        while current.parent_variant:
            chain.append(current.parent_variant)
            current = current.parent_variant
        chain.append(current.source_image)
        return list(reversed(chain))

    @property
    def size(self) -> int | None:
        """Get the size parameter for resized variants."""
        if self.variant_type == 'resized' and self.parameters:
            return self.parameters.get('size')
        return None

    @classmethod
    def find_by_size(cls, session, source_image_id: int, size: int) -> "ImageVariant | None":
        """Find a resized variant of specific size for an image."""
        return (session.query(cls)
                .filter(cls.source_image_id == source_image_id,
                       cls.variant_type == 'resized',
                       cls.parameters['size'].astext.cast(Integer) == size)
                .first())
    @property
    def size(self) -> int | None:
        """Get the size parameter for resized variants."""
        if self.variant_type == 'resized' and self.parameters:
            return self.parameters.get('size')
        return None

    @classmethod
    def find_by_pose(cls, session, source_image_id: int, face: bool) -> "ImageVariant | None":
        """Find a posed variant of specific image."""
        return (session.query(cls)
                .filter(cls.source_image_id == source_image_id,
                       cls.variant_type == 'posed',
                       cls.parameters['face'].astext.cast(Boolean) == face)
                .first())
    @property
    def face(self) -> str | None:
        """Get the face parameter for posed variants."""
        if self.variant_type == 'posed' and self.parameters:
            return self.parameters.get('face')
        return None
    
    @classmethod
    def find_by_segment(cls, session, source_image_id: int, inverse: bool) -> "ImageVariant | None":
        """Find a segmented variant of specific inverse for an image."""
        return (session.query(cls)
                .filter(cls.source_image_id == source_image_id,
                       cls.variant_type == 'segmented',
                       cls.parameters['inverse'].astext.cast(Boolean) == inverse)
                .first())
    @property
    def inverse(self) -> int | None:
        """Get the inverse parameter for segmentedd variants."""
        if self.variant_type == 'segmentedd' and self.parameters:
            return self.parameters.get('inverse')
        return None
    
    @classmethod
    def get_all_variants(cls, session, source_image_id: int, variant_type: str | None = None):
        """Get all variants for an image, optionally filtered by type."""
        query = session.query(cls).filter(cls.source_image_id == source_image_id)
        if variant_type:
            query = query.filter(cls.variant_type == variant_type)
        return query.all() 

    @property
    def id(self):
        """Alias for variant_id for compatibility."""
        return self.variant_id


    @property
    def pose_path(self):
        """Get the pose path for pose variants."""
        if self.variant_type == 'pose' and self.parameters:
            return self.parameters.get('pose_path')
        return None
