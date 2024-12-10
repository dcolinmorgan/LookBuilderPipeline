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
    variant_type = Column(String, nullable=False)
    parameters = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    processed = Column(Boolean, default=False)

    # Composition - reference to source image
    source_image = relationship("Image", back_populates="variants")

    def get_image_data(self, session):
        """Get variant image data using its own oid"""
        if not self.variant_oid:
            return None
            
        try:
            connection = session.connection().connection
            lob = connection.lobject(oid=self.variant_oid, mode='rb')
            data = lob.read()
            lob.close()
            return data
        except Exception as e:
            return None

    @property
    def user(self):
        """Get user through composition"""
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
            return existing_variant

        # Create new variant if none exists
        logging.info(f"Creating new variant of type {variant_type}")
        variant = self.create_variant(
            variant_type,
            session,
            **kwargs
        )
        logging.info(f"Variant created of type {variant_type}")

        return variant


    def get_variant(self, variant_type: str, session, **kwargs):
        """Get a variant with specific parameters"""
        # Merge the image with the current session
        session.add(self)
        
        filter_conditions = [
            ImageVariant.source_image_id == self.source_image_id,
            ImageVariant.variant_type == variant_type
        ]
        
        for key, value in kwargs.items():
            if key == 'size':
                # Handle size as single integer
                filter_conditions.append(
                    cast(ImageVariant.parameters['size'].astext, Integer) == value
                )
            else:
                # Handle other parameters normally
                if isinstance(value, bool):
                    filter_conditions.append(
                        cast(ImageVariant.parameters[key].astext, Boolean) == value
                    )
                elif isinstance(value, float):
                    filter_conditions.append(
                        cast(ImageVariant.parameters[key].astext, Float) == value
                    )
                else:
                    filter_conditions.append(
                        cast(ImageVariant.parameters[key].astext, String) == str(value)
                    )
        
        try:
            return session.query(ImageVariant).filter(*filter_conditions).first()
        finally:
            # Clean up by expunging the image from session
            session.expunge(self)


    def get_variant_image(self, variant: 'self', session) -> Optional[bytes]:
        """Get the actual image data from a variant.
        
        Args:
            variant: self instance to get image from
            session: Database session
            
        Returns:
            bytes: Image data if found, None otherwise
        """
        if not variant:
            logging.error("No variant provided to get_variant_image")
            return None
            
        try:
            logging.info(f"Getting image data for variant {variant.id}")
            connection = session.connection().connection
            
            lob = connection.lobject(oid=variant.variant_oid, mode='rb')
            data = lob.read()
            lob.close()
            
            logging.info(f"Successfully read {len(data)} bytes from variant {variant.id}")
            return data
            
        except Exception as e:
            logging.error(f"Error reading variant image data: {str(e)}", exc_info=True)
            return None

    def create_variant(self, variant_type: str, session, **kwargs) -> 'ImageVariant':
        """Create a new variant of the specified type."""
        try:
            logging.info(f"Creating variant of type {variant_type}")
            
            # Get source image data from the source_image relationship
            source_image = session.merge(self.source_image)  # Use source_image relationship
            if not source_image:
                raise ValueError(f"No source image found for variant (source_image_id={self.source_image_id})")
            
            image_data = source_image.get_image_data(session)
            if not image_data:
                raise ValueError(f"No image data found for source image {source_image.image_id}")

            # Import the appropriate variant class
            if variant_type == 'pose':
                from .pose_variant import PoseVariant
                variant_class = PoseVariant
            elif variant_type == 'segment':
                from .segment_variant import SegmentVariant
                variant_class = SegmentVariant
            else:
                raise ValueError(f"Invalid variant type: {variant_type}")

            # Create and process the variant
            variant = variant_class(
                source_image_id=source_image.image_id,
                variant_type=variant_type,
                parameters=kwargs
            )
            session.add(variant)  # Add to session before processing
            session.flush()  # Ensure variant has an ID
            
            # Get processed image using the appropriate method
            if variant_type == 'pose':
                processed_image = variant.get_pose_image(session)
            elif variant_type == 'segment':
                processed_image = variant.get_segment_image(session)
            
            if processed_image is None:
                raise ValueError(f"Failed to process {variant_type} variant")
            
            # Convert to bytes if needed
            if not isinstance(processed_image, bytes):
                img_byte_arr = io.BytesIO()
                processed_image.save(img_byte_arr, format='PNG')
                processed_image = img_byte_arr.getvalue()
            
            # Store the processed image
            lob = session.connection().connection.lobject(mode='wb')
            lob.write(processed_image)
            variant.variant_oid = lob.oid
            lob.close()
            
            session.flush()
            return variant
            
        except Exception as e:
            logging.error(f"Error in create_variant: {str(e)}")
            raise
