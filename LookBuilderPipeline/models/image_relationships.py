"""Define relationships between Image and its variants."""
from sqlalchemy.orm import relationship
from .image import Image
from .image_variant import ImageVariant

# Image relationships
Image.variants = relationship(
    "ImageVariant", 
    back_populates="source_image",
    foreign_keys="ImageVariant.source_image_id"
)

# ImageVariant relationships
ImageVariant.source_image = relationship(
    "Image", 
    back_populates="variants",
    foreign_keys="ImageVariant.source_image_id"
)

ImageVariant.parent_variant = relationship(
    "ImageVariant", 
    remote_side="ImageVariant.variant_id",
    backref="child_variants"
)

ImageVariant.user = relationship("User")