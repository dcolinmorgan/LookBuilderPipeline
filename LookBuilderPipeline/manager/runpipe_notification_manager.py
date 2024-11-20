from datetime import datetime
import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux 

class RunPipeNotificationManager(NotificationManager):
    """Handles image load pipeline operations."""
    
    def __init__(self):
        super().__init__()
        self.channels = ['gen_image]
        self.required_fields = ['process_id', 'image_id', 'prompt']
        logging.info(f"LoadPipelineNotificationManager listening on channels: {self.channels}")

    def process_item(self, load_request):
        """Process a single load request."""
        return self.process_run({
            'process_id': load_request.process_id,
            'image_id': load_request.image_id,
            'prompt': load_request.parameters.get('prompt', None)
        })

    def process_run(self, load_data):
        """Process a load request through its stages."""
        logging.info(f"Processing load request: {load_data}")
        
        try:
            validated_data = self.validate_process_data(load_data)
            process_id = validated_data['process_id']
            
            def execute_load_process(session):
                # Get the image from database
                image = session.query(Image).get(validated_data['image_id'])
                if not image:
                    raise ValueError(f"Image {validated_data['image_id']} not found")

                # Initiate model generation
                self.variant, self.save_path = self.model.generate_image(prompt=validated_data.get('prompt', None))
                
                if not self.variant:
                    raise ValueError("Failed to generate image variant")
                    
                return self.variant.id
            
            return self.process_with_error_handling(process_id, execute_load_process)
            
        except Exception as e:
            logging.error(f"Error processing loadpipe: {str(e)}", exc_info=True)
            raise

    def handle_notification(self, channel, data):
        """Handle loadpipe notifications."""
        logging.info(f"LoadPipelineNotificationManager received: channel={channel}, data={data}")
        
        if channel == 'gen_image':
            return self.process_run(data)
            
        logging.warning(f"Unexpected channel: {channel}")
        return None
