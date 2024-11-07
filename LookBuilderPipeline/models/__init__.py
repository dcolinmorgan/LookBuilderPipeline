from .base import Base
from .user import User
from .image import Image
from .image_variant import ImageVariant
from .image_relationships import setup_relationships

# Set up all relationships after all models are imported
setup_relationships()

__all__ = ['Base', 'User', 'Image', 'ImageVariant'] 