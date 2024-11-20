from datetime import datetime
import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.segment import segment_image

class SegmentNotificationManager(NotificationManager):
    """Handles image segment operations."""
    
    def __init__(self):
        super().__init__()
        self.channels = ['segment_image']
        self.required_fields = ['process_id', 'image_id', 'target_size']
        logging.info(f"SegmentNotificationManager listening on channels: {self.channels}")

    def process_item(self, segment_request):
        """Process a single segment request."""
        return self.process_segment({
            'process_id': segment_request.process_id,
            'image_id': segment_request.image_id,
            'inverse': segment_request.parameters.get('inverse', True)
        })

    def process_segment(self, segment_data):
        """Process a segment request through its stages."""
        logging.info(f"Processing segment request: {segment_data}")
        
        try:
            validated_data = self.validate_process_data(segment_data)
            process_id = validated_data['process_id']
            
            def execute_segment_process(session):
                # Get the image from database
                image = session.query(Image).get(validated_data['image_id'])
                if not image:
                    raise ValueError(f"Image {validated_data['image_id']} not found")

                # Get image data
                image_data = image.get_image_data(session)
                if not image_data:
                    raise ValueError("No image data found")

                # Process the segment using segment function
                self.segment_variant = segment_image(
                    image_data,
                    inverse=validated_data.get('inverse', True)
                )
                
                
                if not self.segment_variant:
                    raise ValueError("Failed to create segment variant")
                    
                return self.segment_variant.id
            
            return self.process_with_error_handling(process_id, execute_segment_process)
            
        except Exception as e:
            logging.error(f"Error processing segment: {str(e)}", exc_info=True)
            raise

    def handle_notification(self, channel, data):
        """Handle segment notifications."""
        logging.info(f"SegmentNotificationManager received: channel={channel}, data={data}")
        
        if channel == 'segment_image':
            return self.process_segment(data)
            
        logging.warning(f"Unexpected channel: {channel}")
        return None
