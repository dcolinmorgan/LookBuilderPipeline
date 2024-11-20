from datetime import datetime
import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux 

class LoadPipeNotificationManager(NotificationManager):
    """Handles image load pipeline operations."""
    
    def __init__(self):
        super().__init__()
        self.channels = ['load_pipeline']
        self.required_fields = ['process_id', 'image_id', 'model_type']
        logging.info(f"LoadPipelineNotificationManager listening on channels: {self.channels}")

    def process_item(self, load_request):
        """Process a single load request."""
        return self.process_load({
            'process_id': load_request.process_id,
            'image_id': load_request.image_id,
            'model_type': load_request.parameters.get('model_type', 'sdxl')
        })

    def process_load(self, load_data):
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

                # Initialize appropriate model
                model_type = validated_data['model_type'].lower()
                if model_type == 'sdxl':
                    self.model = ImageModelSDXL.prepare_model()
                elif model_type == 'flux':
                    self.model = ImageModelFlux.prepare_model()
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                self.model.prepare_image(self.image.id, self.pose_variant.id, self.segment_variant.id)
                
                if not self.model:
                    raise ValueError("Failed to create loadpipe variant")
                    
                return variant.id
            
            return self.process_with_error_handling(process_id, execute_load_process)
            
        except Exception as e:
            logging.error(f"Error processing loadpipe: {str(e)}", exc_info=True)
            raise

    def handle_notification(self, channel, data):
        """Handle loadpipe notifications."""
        logging.info(f"LoadPipelineNotificationManager received: channel={channel}, data={data}")
        
        if channel == 'load_pipe':
            return self.process_load(data)
            
        logging.warning(f"Unexpected channel: {channel}")
        return None
