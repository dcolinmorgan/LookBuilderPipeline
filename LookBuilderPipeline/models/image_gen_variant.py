from typing import Optional
from .image_variant import ImageVariant
from LookBuilderPipeline.image_generation.image_generation_sdxl import ImageGenerationSDXL
from LookBuilderPipeline.image_generation.image_generation_fl2 import ImageGenerationFlux
from LookBuilderPipeline.models.image import Image
import logging
from abc import ABC, abstractmethod

class ModelFactory:
    """Factory for creating model instances"""
    _models = {
        'sdxl': ImageGenerationSDXL,
        'flux': ImageGenerationFlux
    }
    
    @classmethod
    def register_model(cls, model_type: str, model_class):
        cls._models[model_type] = model_class
    
    @classmethod
    def create(cls, model_type: str, **kwargs):
        model_class = cls._models.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class(**kwargs)

class ImageGenVariant(ImageVariant):
    """Base class for generation variants."""
    __mapper_args__ = {
        'polymorphic_identity': 'image_gen_variant'
    }

    variant_class = ('LookBuilderPipeline.models.image_gen_variant', 'ImageGenVariant')
    
    def __init__(self, source_image_id=None, variant_type=None, parameters=None, **kwargs):
        super().__init__(
            source_image_id=source_image_id,
            variant_type=variant_type,
            parameters=parameters,
            **kwargs
        )
        self.model_factory = ModelFactory()

    def process_image(self, session) -> Optional[bytes]:
        """Process image using appropriate model."""
        try:
            if not self.source_image:
                raise ValueError(f"No source image found for ID {self.source_image_id}")
                
            image_data = self.source_image.get_image_data(session)
            if not image_data:
                raise ValueError("No source image data found")
            
            # Create appropriate model using factory
            model = self.model_factory.create(
                model_type=self.parameters['model_type'],
                image=image_data,
                prompt=self.parameters['prompt'],
                negative_prompt=self.parameters.get('negative_prompt', ''),
                seed=self.parameters.get('seed'),
                strength=self.parameters.get('strength', 1.0),
                guidance_scale=self.parameters.get('guidance_scale', 7.5),
                # LoRA=self.parameters.get('LoRA')
            )
            
            # Process image using selected model
            # model.prepare_model()
            return model.generate_image()
            
        except Exception as e:
            logging.error(f"Error creating variant: {str(e)}")
            raise
    

    # @property
    # def id(self):
    #     """Alias for variant_id for compatibility."""
    #     return self.variant_id

