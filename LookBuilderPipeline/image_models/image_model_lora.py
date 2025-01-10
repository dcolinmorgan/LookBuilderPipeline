import logging
import os
import requests
from huggingface_hub import hf_hub_download
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux
from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL


class ImageModelLORA:
    """Factory class to create appropriate LoRA model
    demo code for how to use this
    # Create SDXL LoRA model
    lora_model = ImageModelLORA.create('sdxl', 
    image=input_image, 
    LoRA='supermodel_face',
    prompt="your prompt",
    negative_prompt="your negative prompt",
    # other SDXL parameters...)
    
    # Prepare model (this will set up SDXL and add LoRA weights)
    lora_model.prepare_model()

    # Generate image using inherited method
    generated_image = lora_model.generate_image()"""
    @staticmethod
    def create(model_type, image, **kwargs):
        if model_type.lower() == 'sdxl':
            return SDXLLoRA(image, **kwargs)
        elif model_type.lower() == 'flux':
            return FluxLoRA(image, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class SDXLLoRA(ImageModelSDXL):
    def __init__(self, image, **kwargs):
        super().__init__(image, **kwargs)
        self.lora_type = kwargs.get('LoRA')
        self.lora_weight = kwargs.get('lora_weight', 1.0)

    def prepare_model(self):
        """Prepare SDXL model and add LoRA weights"""
        super().prepare_model()  # Initialize base SDXL model first
        
        if not self.lora_type:
            return

        # LoRA model IDs
        lora_ids = {
            'supermodel_face': 231666,
            'female_face': 273591,
            'better_face': 301988,
            'diana': 293406
        }

        try:
            if self.lora_type in lora_ids:
                # Download and load CivitAI LoRA
                self._download_civitai_lora(lora_ids[self.lora_type])
                self.pipe.load_lora_weights(
                    'LookBuilderPipeline/LookBuilderPipeline/image_models',
                    weight_name=f'{self.lora_type}.safetensors',
                    adapter_name=self.lora_type
                )
            elif self.lora_type == 'mg1c':
                # Load Hugging Face LoRA
                self.pipe.load_lora_weights(
                    "Dcolinmorgan/style-mg1c",
                    token=os.getenv("HF_SD3_FLUX"),
                    adapter_name=self.lora_type
                )
            elif self.lora_type == 'hyper':
                # Load Hyper-SD LoRA
                lora_path = hf_hub_download(
                    "ByteDance/Hyper-SD",
                    "Hyper-SDXL-2steps-lora.safetensors"
                )
                self.pipe.load_lora_weights(lora_path, adapter_name=self.lora_type)
                self.pipe.scheduler.timestep_spacing = "trailing"

            # Activate LoRA weights
            if self.lora_type:
                logging.info(f"Activating LoRA: {self.lora_type}")
                self.pipe.set_adapters(self.lora_type, adapter_weights=[self.lora_weight])

        except Exception as e:
            logging.error(f"Error loading LoRA weights: {str(e)}")
            raise

    def _download_civitai_lora(self, lora_id):
        """Download LoRA model from CivitAI"""
        url = f'https://civitai.com/api/download/models/{lora_id}'
        response = requests.get(url)
        if response.status_code == 200:
            fname = f'/LookBuilderPipeline/LookBuilderPipeline/image_models/{self.lora_type}.safetensors'
            with open(fname, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download LoRA model: {response.status_code}")


class FluxLoRA(ImageModelFlux):
    def __init__(self, image, **kwargs):
        super().__init__(image, **kwargs)
        self.lora_type = kwargs.get('LoRA')
        self.lora_weight = kwargs.get('lora_weight', 1.0)

    def prepare_model(self):
        """Prepare Flux model and add LoRA weights"""
        super().prepare_model()  # Initialize base Flux model first
        
        # Add Flux-specific LoRA handling here
        if self.lora_type:
            logging.info(f"Activating Flux LoRA: {self.lora_type}")
            # Add Flux LoRA implementation when ready
