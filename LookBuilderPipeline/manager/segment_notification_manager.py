import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
import select

class SegmentNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing SegmentNotificationManager")
        self.channels = ['image_segment']
        logging.info(f"SegmentNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id', 'inverse']

    def handle_notification(self, channel, data):
        """Handle segment notifications."""
        logging.info(f"SegmentNotificationManager received: channel={channel}, data={data}")
        if channel == 'image_segment':
            # Get the full process data from ProcessQueue
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Merge the notification data with process parameters
                    full_data = {
                        'process_id': data['process_id'],
                        'image_id': data['image_id'],
                        'inverse': process.parameters.get('inverse')
                    }
                    logging.info(f"Processing segment with parameters: {full_data}")
                    return self.process_segment(full_data)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"SegmentNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, segment):
        """Process a single segment notification."""
        return self.process_segment({
            'process_id': segment.process_id,
            'image_id': segment.image_id,
            'inverse': segment.parameters.get('inverse')
        })

    def process_segment(self, segment_data):
        """Process a segment notification through its stages."""
        validated_data = self.validate_process_data(segment_data)
        process_id = validated_data['process_id']
        
        def execute_segment_process(session):
            # Get the image
            image = session.query(Image).get(validated_data['image_id'])
            if not image:
                raise ValueError(f"Image {validated_data['image_id']} not found")

            # Create segment variant
            variant = image.get_or_create_segment_variant(
                session=self.session,
                inverse=validated_data['inverse']
            )
            
            if not variant:
                raise ValueError("Failed to create SDXL variant")
            
            return variant.id
        
        return self.process_with_error_handling(process_id, execute_segment_process)
