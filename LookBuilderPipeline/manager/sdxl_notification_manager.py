import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant


class SDXLNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing SDXLNotificationManager")
        self.channels = ['image_sdxl']
        self.required_fields = ['process_id', 'image_id']
        logging.info(f"SDXLNotificationManager listening on channels: {self.channels}")

    def handle_notification(self, channel, data):
        """Handle SDXL notifications."""
        logging.info(f"SDXLNotificationManager received: channel={channel}, data={data}")
        
        if channel == 'image_sdxl':
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Get parameters from process.parameters
                    full_data = {
                        'process_id': data['process_id'],
                        'image_id': data['image_id'],
                        'prompt': process.parameters.get('prompt'),
                        'negative_prompt': process.parameters.get('negative_prompt', 'ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves'),
                        'seed': process.parameters.get('seed', None),  # Optional seed
                        'strength': process.parameters.get('strength', 1.0),  # Default to 1.0
                        'guidance_scale': process.parameters.get('guidance_scale', 7.5),  # Default to 7.5
                        'LoRA': process.parameters.get('LoRA', None)
                    }
                    logging.info(f"Processing SDXL with parameters: {full_data}")
                    return self.process_sdxl(full_data)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
            
        logging.warning(f"Unexpected channel: {channel}")
        return None

    def process_item(self, sdxl_request):
        """Process a single SDXL request."""
        return self.process_sdxl({
            'process_id': sdxl_request.process_id,
            'image_id': sdxl_request.image_id,
            'prompt': sdxl_request.parameters.get('prompt'),
            'negative_prompt': sdxl_request.parameters.get('negative_prompt', 'ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves'),
            'seed': sdxl_request.parameters.get('seed', None),
            'strength': sdxl_request.parameters.get('strength', 1.0),
            'guidance_scale': sdxl_request.parameters.get('guidance_scale', 7.5),
            'LoRA': sdxl_request.parameters.get('LoRA', None)
        })

    def process_sdxl(self, sdxl_data):
        """Process an SDXL request."""
        validated_data = self.validate_process_data(sdxl_data)
        process_id = validated_data['process_id']
        
        def execute_sdxl_process(session):
            # Get the image
            image = session.query(Image).get(validated_data['image_id'])
            if not image:
                raise ValueError(f"Image {validated_data['image_id']} not found")
            
            # Create a temporary ImageVariant instance to use get_or_create_variant
            base_variant = ImageVariant(
                source_image_id=image.image_id,
                variant_type='sdxl'
            )
            session.add(base_variant)
            session.flush()
            
            # Create SDXL variant
            variant = base_variant.get_or_create_variant(
                session=session,
                variant_type='sdxl',
                prompt=validated_data['prompt'],
                negative_prompt=validated_data['negative_prompt'],
                seed=validated_data['seed'],
                strength=validated_data['strength'],
                guidance_scale=validated_data['guidance_scale'],
                LoRA=validated_data['LoRA']
            )
            
            if not variant:
                raise ValueError("Failed to create SDXL variant")
                
            return variant.id
        
        return self.process_with_error_handling(process_id, execute_sdxl_process)
