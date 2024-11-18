"""Define relationships between Image and its variants."""
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, DateTime, ForeignKey, Table
from sqlalchemy.sql import func
from .base import Base

look_images = Table(
    'look_images',
    Base.metadata,
    Column('look_id', Integer, ForeignKey('looks.look_id', ondelete='CASCADE'), primary_key=True),
    Column('image_id', Integer, ForeignKey('images.image_id', ondelete='CASCADE'), primary_key=True),
    Column('created_at', DateTime, server_default=func.now())
)

# Image to ImageVariant relationship
def setup_relationships():
    from .image import Image
    from .image_variant import ImageVariant
    from .user import User
    from .look import Look
    
    # Existing relationships
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
        primaryjoin="Image.user_id == User.id",
        lazy="dynamic"
    )

    Image.user = relationship(
        "User",
        back_populates="images",
        primaryjoin="Image.user_id == User.id"
    )

    # Look relationships with explicit join conditions
    Look.user = relationship(
        "User",
        back_populates="looks",
        primaryjoin="Look.user_id == User.id"
    )
    
    User.looks = relationship(
        "Look",
        back_populates="user",
        primaryjoin="Look.user_id == User.id",
        cascade="all, delete"
    )

    # Image-Look relationships
    Image.looks = relationship(
        "Look",
        secondary=look_images,
        back_populates="images"
    )
    
    Look.images = relationship(
        "Image",
        secondary=look_images,
        back_populates="looks"
    )

# Call setup at the end of __init__.py