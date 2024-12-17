from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
from LookBuilderPipeline.models.image import Image
import logging
import io

class SDXLVariant(ImageVariant):
    """A specialized variant class for sdxl transformations."""
    __mapper_args__ = {
        'polymorphic_identity': 'sdxl_variant'
    }

    variant_class = ('LookBuilderPipeline.models.sdxl_variant', 'SDXLVariant')
    
    @property
    def sdxl_parameters(self) -> dict:
        """Get the sdxl parameters."""
        return self.parameters.get('sdxl', {})


    def __init__(self, source_image_id=None, variant_type=None, parameters=None, **kwargs):
        """Initialize sdxl variant"""
        super().__init__(
            source_image_id=source_image_id,
            variant_type=variant_type,
            parameters=parameters,
            **kwargs
        )

    def process_image(self, session) -> Optional[bytes]:
        """Use original image to create sdxl variant."""
        try:
            # Get source image data properly through session
            if not self.source_image:
                raise ValueError(f"No source image found for ID {self.source_image_id}")
                
            image_data = self.source_image.get_image_data(session)
            if not image_data:
                raise ValueError("No source image data found")
            
            # Create image model and get sdxl
            image_model = ImageModelSDXL(image=image_data, prompt=self.parameters['prompt'], negative_prompt=self.parameters['negative_prompt'], seed=self.parameters['seed'], strength=self.parameters['strength'], guidance_scale=self.parameters['guidance_scale'], LoRA=self.parameters['LoRA'])
            image_model.prepare_model()
            generated_image = image_model.generate_image()
            
            return generated_image
            
        except Exception as e:
            logging.error(f"Error creating segment variant: {str(e)}")
            raise
    

    # @property
    # def id(self):
    #     """Alias for variant_id for compatibility."""
    #     return self.variant_id

