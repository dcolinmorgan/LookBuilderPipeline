from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.pose import detect_pose
from LookBuilderPipeline.models.image import Image
import logging
import io

class PoseVariant(ImageVariant):
    """A specialized variant class for pose transformations."""
    __mapper_args__ = {
        'polymorphic_identity': 'pose_variant'
    }

    variant_class = ('LookBuilderPipeline.models.pose_variant', 'PoseVariant')

    @property
    def pose_parameters(self) -> dict:
        """Get the pose parameters."""
        return self.parameters.get('pose', {})

    @classmethod
    def find_by_pose(cls, session, source_image_id: int, pose_parameters: dict) -> Optional["PoseVariant"]:
        """Find a pose variant with specific pose parameters for an image."""
        return (session.query(cls)
                .filter(cls.source_image_id == source_image_id,
                       cls.parameters['pose'].astext == pose_parameters)
                .first()) 

    def __init__(self, source_image_id=None, variant_type=None, parameters=None, **kwargs):
        """Initialize pose variant"""
        super().__init__(
            source_image_id=source_image_id,
            variant_type=variant_type,
            parameters=parameters,
            **kwargs
        )


    def process_image(self, session) -> Optional[bytes]:
        """Use original image to create pose variant."""
        try:
            # Get source image data properly through session
            if not self.source_image:
                raise ValueError(f"No source image found for ID {self.source_image_id}")
                
            image_data = self.source_image.get_image_data(session)
            if not image_data:
                raise ValueError("No source image data found")

            # Create image model and get pose
            pose_image = detect_pose(image_data, face=self.pose_parameters.get('face'))
            
            return pose_image
            
        except Exception as e:
            logging.error(f"Error creating pose variant: {str(e)}")
            raise
    

    # @property
    # def id(self):
    #     """Alias for variant_id for compatibility."""
    #     return self.variant_id

