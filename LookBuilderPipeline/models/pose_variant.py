from typing import Optional, Dict
from .image_variant import ImageVariant

class PoseVariant(ImageVariant):
    """A specialized variant class for pose transformations."""
    
    __mapper_args__ = {
        'polymorphic_identity': 'pose_variant'
    }

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