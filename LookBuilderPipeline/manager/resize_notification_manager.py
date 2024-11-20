from datetime import datetime
import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.utils.resize import resize_images

class ResizeNotificationManager(NotificationManager):
    """Handles image resize operations."""
    
    def __init__(self):
        super().__init__()
        self.channels = ['resize_image']
        self.required_fields = ['process_id', 'image_id', 'target_size']
        logging.info(f"ResizeNotificationManager listening on channels: {self.channels}")

    def process_item(self, resize_request):
        """Process a single resize request."""
        return self.process_resize({
            'process_id': resize_request.process_id,
            'image_id': resize_request.image_id,
            'target_size': resize_request.parameters.get('target_size', 512),
            'aspect_ratio': resize_request.parameters.get('aspect_ratio'),
            'square': resize_request.parameters.get('square', False)
        })

    def process_resize(self, resize_data):
        """Process a resize request through its stages."""
        logging.info(f"Processing resize request: {resize_data}")
        
        try:
            validated_data = self.validate_process_data(resize_data)
            process_id = validated_data['process_id']
            
            def execute_resize_process(session):
                # Get the image from database
                image = session.query(Image).get(validated_data['image_id'])
                if not image:
                    raise ValueError(f"Image {validated_data['image_id']} not found")

                # Get image data
                image_data = image.get_image_data(session)
                if not image_data:
                    raise ValueError("No image data found")

                # Process the resize using resize_images function
                self.resized_variant = resize_images(
                    image_data,
                    target_size=validated_data['target_size'],
                    aspect_ratio=validated_data.get('aspect_ratio', 1.0),
                    square=validated_data.get('square', False)
                )
                
                if not self.resized_variant:
                    raise ValueError("Failed to create resize variant")
                    
                return self.resized_variant.id
            
            return self.process_with_error_handling(process_id, execute_resize_process)
            
        except Exception as e:
            logging.error(f"Error processing resize: {str(e)}", exc_info=True)
            raise

    def handle_notification(self, channel, data):
        """Handle resize notifications."""
        logging.info(f"ResizeNotificationManager received: channel={channel}, data={data}")
        
        if channel == 'resize_image':
            return self.process_resize(data)
            
        logging.warning(f"Unexpected channel: {channel}")
        return None
