from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from LookBuilderPipeline.models.image import Image
import logging
import io

class ResizeVariant(ImageVariant):
    """Resize-specific variant implementation"""
    __mapper_args__ = {
        'polymorphic_identity': 'resize'
    }

    def __init__(self, source_image_id=None, variant_type=None, parameters=None, **kwargs):
        """Initialize resize variant"""
        super().__init__(
            source_image_id=source_image_id,
            variant_type=variant_type,
            parameters=parameters,
            **kwargs
        )

    def process_image(self, session):
        """Process the resize image after initialization"""
        logging.info("Processing resize variant")
        try:
            # Process the resize
            processed_image = self.get_resize_image(session)
            if processed_image:
                # Store the processed image
                lob = session.connection().connection.lobject(mode='wb')
                lob.write(processed_image)
                self.variant_oid = lob.oid
                lob.close()
                self.processed = True
                logging.info(f"Successfully processed resize variant {self.variant_id}")
        except Exception as e:
            logging.error(f"Failed to process resize variant: {str(e)}")
            raise

    def get_resize_image(self, session) -> Optional[bytes]:
        """Get the resize image for this resize variant."""
        try:
            # Get source image data properly through session
            source_image = session.query(Image).get(self.source_image_id)
            if not source_image:
                raise ValueError(f"No source image found for ID {self.source_image_id}")
                
            image_data = source_image.get_image_data(session)
            if not image_data:
                raise ValueError("No source image data found")

            # Create image model and get resize
            image_model = BaseImageModel(image=image_data)
            resize_img = image_model.image
            
            # Convert to bytes if needed
            if not isinstance(resize_img, bytes):
                img_byte_arr = io.BytesIO()
                resize_img.save(img_byte_arr, format='PNG')
                return img_byte_arr.getvalue()
            
            return resize_img
            
        except Exception as e:
            logging.error(f"Error creating resize variant: {str(e)}")
            raise
    

    # @property
    # def id(self):
    #     """Alias for variant_id for compatibility."""
    #     return self.variant_id

