import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant


class GenNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing GenNotificationManager")
        self.channels = ['image_gen']
        self.required_fields = ['process_id', 'image_id', 'model_type']
        logging.info(f"GenNotificationManager listening on channels: {self.channels}")

    def handle_notification(self, channel, data):
        """Handle GEN notifications."""
        logging.info(f"GenNotificationManager received: channel={channel}, data={data}")
        
        if channel == 'image_gen':
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    if 'model_type' not in process.parameters:
                        logging.error("No model_type specified in parameters")
                        return None
                    return self.process_item(process)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
            
        logging.warning(f"Unexpected channel: {channel}")
        return None

    def process_item(self, gen_request):
        """Process a single GEN request."""
        validated_data = {
            'process_id': gen_request.process_id,
            'image_id': gen_request.image_id,
            'model_type': gen_request.parameters.get('model_type','sdxl'),
            'prompt': gen_request.parameters.get('prompt'),
            'negative_prompt': gen_request.parameters.get('negative_prompt', ''),
            'seed': gen_request.parameters.get('seed', None),
            'strength': gen_request.parameters.get('strength', 1.0),
            'guidance_scale': gen_request.parameters.get('guidance_scale', 7.5),
            'lora_type': gen_request.parameters.get('LoRA', None)
        }

    # def process_gen(self, gen_data):
        # """Process an GEN request."""
        # validated_data = self.validate_process_data(gen_data)
        process_id = validated_data['process_id']
        
        def execute_gen_process(session):
            # Get the image
            image = session.query(Image).get(validated_data['image_id'])
            if not image:
                raise ValueError(f"Image {validated_data['image_id']} not found")
            
            model_type = validated_data['model_type']
            
            # Create a temporary ImageVariant instance to use get_or_create_variant
            base_variant = ImageVariant(
                source_image_id=image.image_id,
                variant_type= 'image_gen'
            )
            session.add(base_variant)
            session.flush()
            
            # Create GEN variant
            variant = base_variant.get_or_create_variant(
                session=session,
                variant_type='image_gen',
                model_type=model_type,
                prompt=validated_data['prompt'],
                # negative_prompt=validated_data['negative_prompt'],
                seed=validated_data['seed'],
                strength=validated_data['strength'],
                guidance_scale=validated_data['guidance_scale'],
                # LoRA=validated_data['LoRA']
            )
            
            if not variant:
                raise ValueError("Failed to create image variant")
                
            return variant.id
        
        return self.process_with_error_handling(process_id, execute_gen_process)
