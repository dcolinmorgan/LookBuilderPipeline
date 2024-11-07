"""Define relationships between Image and its variants."""
from sqlalchemy.orm import relationship

# Image to ImageVariant relationship
def setup_relationships():
    from .image import Image
    from .image_variant import ImageVariant
    from .user import User
    
    # Image-ImageVariant relationships
    Image.variants = relationship(
        "ImageVariant",
        back_populates="source_image",
        foreign_keys="ImageVariant.source_image_id",
        lazy="dynamic"
    )

    ImageVariant.source_image = relationship(
        "Image",
        back_populates="variants",
        foreign_keys="ImageVariant.source_image_id"
    )

    # User-Image relationship with explicit foreign key
    User.images = relationship(
        "Image",
        back_populates="user",
        lazy="dynamic"
    )

    Image.user = relationship(
        "User",
        back_populates="images"
    )

# Call setup at the end of __init__.py