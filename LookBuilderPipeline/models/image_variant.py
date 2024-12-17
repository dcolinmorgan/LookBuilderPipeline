from typing import Optional, List, Union, TYPE_CHECKING
from datetime import datetime
from .base import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
import logging
import io

class ImageVariant(Base):
    __tablename__ = 'image_variants'
    
    variant_id = Column(Integer, primary_key=True)
    variant_oid = Column(Integer, nullable=True)
    source_image_id = Column(Integer, ForeignKey('images.image_id'), nullable=False)
    parent_variant_id = Column(Integer, ForeignKey('image_variants.variant_id'), nullable=True)
    variant_type = Column(String, nullable=False)
    parameters = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    processed = Column(Boolean, default=False)
    class_type  = Column(String, nullable=False, default='image_variant')

    # Composition - reference to source image
    source_image = relationship("Image", back_populates="variants")

    @property
    def user(self):
        """Get user through the source image."""
        return self.source_image.user if self.source_image else None

    def __repr__(self):
        return f"<self(variant_id={self.variant_id}, type={self.variant_type})>"
 
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

    def get_or_create_variant(self, session, variant_type: str, **kwargs):
        """Get an existing variant or create a new one if it doesn't exist."""
        # Look for existing variant with matching parameters
        existing_variant = (
            session.query(ImageVariant)
            .filter_by(
                source_image_id=self.source_image_id,
                variant_type=variant_type,
                parameters=kwargs
            )
            .first()
        )

        if existing_variant:
            logging.info(f"Found existing variant of type {variant_type} with ID: {existing_variant.variant_id}")
            return existing_variant

        # Create new variant if none exists
        variant = self._create_variant_instance(session, variant_type, **kwargs)
        logging.info(f"Created new variant of type {variant_type} with ID: {variant.variant_id}")
        return variant

    def _create_variant_instance(self, session, variant_type: str, **kwargs) -> 'ImageVariant':
        """Internal method to create a new variant instance."""
        try:
            from .pose_variant import PoseVariant
            from .segment_variant import SegmentVariant
            # from .outfit_variant import OutfitVariant
            from .sdxl_variant import SDXLVariant
            # from .flux_variant import FluxVariant
            # Get the variant class from the registry
            variant_classes = {
                'pose': PoseVariant.variant_class,
                'segment': SegmentVariant.variant_class,
                # 'outfit': OutfitVariant.variant_class,
                'sdxl': SDXLVariant.variant_class,
                # 'flux': FluxVariant.variant_class,
            }

            if variant_type not in variant_classes:
                raise ValueError(f"Invalid variant type: {variant_type}")

            module_path, class_name = variant_classes[variant_type]
            logging.info(f"Importing {class_name} from {module_path}")
            
            # Use importlib for more reliable & dynamic imports without hardcoding.
            # this dynamic import allows the application to handle different types of image variants.
            # the specific variant class is determined at runtime based on the variant_type provided.
            import importlib
            module = importlib.import_module(module_path)
            variant_class = getattr(module, class_name)
            logging.info(f"Imported {variant_class}")
            
            # Create variant instance
            variant = variant_class(
                source_image_id=self.source_image_id,
                variant_type=variant_type,
                parameters=kwargs
            )
            session.add(variant)
            ### we use session.flush() to ensure the temporary base variant is properly initialized before using it to create the actual variant, while still maintaining transaction control.
            ### session.flush() synchronizes the in-memory state of objects with the database, but without committing the transaction. Here's a detailed explanation:
            session.flush()
            
            # Now process the image via the variant class
            processed_image = variant.process_image(session)
            self.source_image.store_image(session, processed_image, variant)

        except Exception as e:
            logging.error(f"Failed to store image or variant: {str(e)}")
            raise
        return variant
            