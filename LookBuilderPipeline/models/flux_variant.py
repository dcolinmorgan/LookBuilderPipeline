from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux
from LookBuilderPipeline.models.image import Image
import logging
import io

class FluxVariant(ImageVariant):
    """Flux-specific variant implementation"""
    __mapper_args__ = {
        'polymorphic_identity': 'flux'
    }

    def __init__(self, source_image_id=None, variant_type=None, parameters=None, **kwargs):
        """Initialize flux variant"""
        super().__init__(
            source_image_id=source_image_id,
            variant_type=variant_type,
            parameters=parameters,
            **kwargs
        )

    def process_image(self, session):
        """Process the flux image after initialization"""
        logging.info("Processing flux variant")
        try:
            # Process the flux
            processed_image = self.get_flux_image(session)
            if processed_image:
                # Store the processed image
                lob = session.connection().connection.lobject(mode='wb')
                lob.write(processed_image)
                self.variant_oid = lob.oid
                lob.close()
                self.processed = True
                logging.info(f"Successfully processed flux variant {self.variant_id}")
        except Exception as e:
            logging.error(f"Failed to process flux variant: {str(e)}")
            raise

    def get_flux_image(self, session) -> Optional[bytes]:
        """Get the flux image for this flux variant."""
        try:
            # Get source image data properly through session
            source_image = session.query(Image).get(self.source_image_id)
            if not source_image:
                raise ValueError(f"No source image found for ID {self.source_image_id}")
                
            image_data = source_image.get_image_data(session)
            if not image_data:
                raise ValueError("No source image data found")

            # Create image model and get flux
            image_model = ImageModelFlux(image=image_data)
            image_model.prepare_model()
            gen_image = image_model.generate_image()
            
            # Convert to bytes if needed
            if not isinstance(gen_image, bytes):
                img_byte_arr = io.BytesIO()
                gen_image.save(img_byte_arr, format='PNG')
                return img_byte_arr.getvalue()
            
            return gen_image
            
        except Exception as e:
            logging.error(f"Error creating flux variant: {str(e)}")
            raise
    

    # @property
    # def id(self):
    #     """Alias for variant_id for compatibility."""
    #     return self.variant_id

