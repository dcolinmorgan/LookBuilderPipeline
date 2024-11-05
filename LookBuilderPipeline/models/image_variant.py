from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import OID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base

class ImageVariant(Base):
    __tablename__ = 'image_variants'
    
    variant_id = Column(Integer, primary_key=True)
    source_image_id = Column(Integer, ForeignKey('images.image_id'), nullable=False)
    parent_variant_id = Column(Integer, ForeignKey('image_variants.variant_id'))
    variant_oid = Column(OID, nullable=False)
    variant_type = Column(String(20), nullable=False)
    parameters = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    processed = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)

    # Relationships
    source_image = relationship("Image", back_populates="variants")
    parent_variant = relationship("ImageVariant", 
                                remote_side=[variant_id],
                                backref="child_variants")
    user = relationship("User", back_populates="image_variants")

    def __repr__(self):
        return f"<ImageVariant(id={self.variant_id}, type={self.variant_type}, source_image={self.source_image_id})>"

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
    @classmethod
    def get_all_variants(cls, session, source_image_id: int, variant_type: str | None = None):
        """Get all variants for an image, optionally filtered by type."""
        query = session.query(cls).filter(cls.source_image_id == source_image_id)
        if variant_type:
            query = query.filter(cls.variant_type == variant_type)
        return query.all() 
