import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
import select

class PoseNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing PoseNotificationManager")
        self.channels = ['image_pose']
        logging.info(f"PoseNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id', 'face']

    def handle_notification(self, channel, data):
        """Handle pose notifications."""
        logging.info(f"PoseNotificationManager received: channel={channel}, data={data}")
        if channel == 'image_pose':
            # Get the full process data from ProcessQueue
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Merge the notification data with process parameters
                    full_data = {
                        'process_id': data['process_id'],
                        'image_id': data['image_id'],
                        'face': process.parameters.get('face')
                    }
                    logging.info(f"Processing pose with parameters: {full_data}")
                    return self.process_pose(full_data)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"PoseNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, pose):
        """Process a single pose notification."""
        return self.process_pose({
            'process_id': pose.process_id,
            'image_id': pose.image_id,
            'face': pose.parameters.get('face')
        })

    def process_pose(self, pose_data):
        """Process a pose notification through its stages."""
        validated_data = self.validate_process_data(pose_data)
        process_id = validated_data['process_id']
        
        def execute_pose_process(session):
            # Get the image
            image = session.query(Image).get(validated_data['image_id'])
            if not image:
                raise ValueError(f"Image {validated_data['image_id']} not found")

            # Create pose variant
            variant = image.get_or_create_pose_variant(
                session=self.session,
                face=validated_data['face']
            )
                
            if not variant:
                raise ValueError("Failed to create pose variant")
            
            return variant.id
        
        return self.process_with_error_handling(process_id, execute_pose_process)
