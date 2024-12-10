from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from LookBuilderPipeline.models.image import Image
import logging
import io

class PoseVariant(ImageVariant):
    """Pose-specific variant implementation"""
    __mapper_args__ = {
        'polymorphic_identity': 'pose'
    }

    def __init__(self, source_image_id=None, variant_type=None, parameters=None, **kwargs):
        """Initialize pose variant"""
        super().__init__(
            source_image_id=source_image_id,
            variant_type=variant_type,
            parameters=parameters,
            **kwargs
        )

    def process_image(self, session):
        """Process the pose image after initialization"""
        logging.info("Processing pose variant")
        try:
            # Process the pose
            processed_image = self.get_pose_image(session)
            if processed_image:
                # Store the processed image
                lob = session.connection().connection.lobject(mode='wb')
                lob.write(processed_image)
                self.variant_oid = lob.oid
                lob.close()
                self.processed = True
                logging.info(f"Successfully processed pose variant {self.variant_id}")
        except Exception as e:
            logging.error(f"Failed to process pose variant: {str(e)}")
            raise

    def get_pose_image(self, session) -> Optional[bytes]:
        """Get the pose image for this pose variant."""
        try:
            # Get source image data properly through session
            source_image = session.query(Image).get(self.source_image_id)
            if not source_image:
                raise ValueError(f"No source image found for ID {self.source_image_id}")
                
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

