from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
import logging
import io

class PoseVariant(ImageVariant):
    """Pose-specific variant implementation"""
    __mapper_args__ = {
        'polymorphic_identity': 'pose'
    }

    def get_pose_image(self, session) -> Optional[bytes]:
        """Get the pose image for this pose variant."""
        try:
            # Get source image data properly through session
            source_image = session.merge(self.source_image)
            image_data = source_image.get_image_data(session)
            
            if not image_data:
                raise ValueError("No source image data found")

            # Create image model and get pose
            image_model = BaseImageModel(image=image_data)
            pose_img = image_model.original_pose
            
            # Convert to bytes if needed
            if not isinstance(pose_img, bytes):
                img_byte_arr = io.BytesIO()
                pose_img.save(img_byte_arr, format='PNG')
                return img_byte_arr.getvalue()
            
            return pose_img
            
        except Exception as e:
            logging.error(f"Error creating pose variant: {str(e)}")
            raise
    

    # @property
    # def id(self):
    #     """Alias for variant_id for compatibility."""
    #     return self.variant_id
