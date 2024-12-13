from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
from LookBuilderPipeline.models.image import Image
import logging
import io

class SDXLVariant(ImageVariant):
    """SDXL-specific variant implementation"""
    __mapper_args__ = {
        'polymorphic_identity': 'sdxl_variant'
    }

    def __init__(self, source_image_id=None, variant_type=None, parameters=None, **kwargs):
        """Initialize sdxl variant"""
        super().__init__(
            source_image_id=source_image_id,
            variant_type=variant_type,
            parameters=parameters,
            **kwargs
        )

    def process_image(self, session):
        """Process the sdxl image after initialization"""
        logging.info("Processing sdxl variant")
        try:
            # Process the sdxl
            processed_image = self.get_sdxl_image(session)
            if processed_image:
                # Store the processed image
                lob = session.connection().connection.lobject(mode='wb')
                lob.write(processed_image)
                self.variant_oid = lob.oid
                lob.close()
                self.processed = True
                logging.info(f"Successfully processed sdxl variant {self.variant_id}")
        except Exception as e:
            logging.error(f"Failed to process sdxl variant: {str(e)}")
            raise

    def get_sdxl_image(self, session) -> Optional[bytes]:
        """Get the sdxl image for this sdxl variant."""
        try:
            # Get source image data properly through session
            source_image = session.query(Image).get(self.source_image_id)
            if not source_image:
                raise ValueError(f"No source image found for ID {self.source_image_id}")
                
            image_data = source_image.get_image_data(session)
            if not image_data:
                raise ValueError("No source image data found")

            # Create image model and get sdxl
            image_model = ImageModelSDXL(image=image_data, prompt=self.parameters['prompt'], negative_prompt=self.parameters['negative_prompt'], seed=self.parameters['seed'], strength=self.parameters['strength'], guidance_scale=self.parameters['guidance_scale'], LoRA=self.parameters['LoRA'])
            image_model.prepare_model()
            gen_image = image_model.generate_image()
            
            # Convert to bytes if needed
            if not isinstance(gen_image, bytes):
                img_byte_arr = io.BytesIO()
                gen_image.save(img_byte_arr, format='PNG')
                return img_byte_arr.getvalue()
            
            return gen_image
            
        except Exception as e:
            logging.error(f"Error creating sdxl variant: {str(e)}")
            raise
    

    # @property
    # def id(self):
    #     """Alias for variant_id for compatibility."""
    #     return self.variant_id

