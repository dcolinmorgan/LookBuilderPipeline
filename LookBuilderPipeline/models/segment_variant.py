from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.segment import segment_image
from LookBuilderPipeline.models.image import Image
import logging
import io

class SegmentVariant(ImageVariant):
    """A specialized variant class for segment transformations."""
    __mapper_args__ = {
        'polymorphic_identity': 'segment_variant'
    }

    variant_class = ('LookBuilderPipeline.models.segment_variant', 'SegmentVariant')

    @property
    def segment_parameters(self) -> dict:
        """Get the segment parameters."""
        return self.parameters.get('segment', {})

    @classmethod
    def find_by_segment(cls, session, source_image_id: int, segment_parameters: dict) -> Optional["SegmentVariant"]:
        """Find a segment variant with specific segment parameters for an image."""
        return (session.query(cls)
                .filter(cls.source_image_id == source_image_id,
                       cls.parameters['segment'].astext == segment_parameters)
                .first()) 

    def __init__(self, source_image_id=None, variant_type=None, parameters=None, **kwargs):
        """Initialize segment variant"""
        super().__init__(
            source_image_id=source_image_id,
            variant_type=variant_type,
            parameters=parameters,
            **kwargs
        )


    def process_image(self, session) -> Optional[bytes]:
        """Use original image to create segment variant."""
        try:
            if not self.source_image:
                raise ValueError(f"No source image found for ID {self.source_image_id}")
                
            image_data = self.source_image.get_image_data(session)
            if not image_data:
                raise ValueError("No source image data found")

            # Create image model and get segment
            _, segmented_image = segment_image(image_data)
            
            return segmented_image
            
        except Exception as e:
            logging.error(f"Error creating segment variant: {str(e)}")
            raise
    

    # @property
    # def id(self):
    #     """Alias for variant_id for compatibility."""
    #     return self.variant_id

