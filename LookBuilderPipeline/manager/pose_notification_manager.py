from datetime import datetime
import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.pose import detect_pose

class PoseNotificationManager(NotificationManager):
    """Handles image pose operations."""
    
    def __init__(self):
        super().__init__()
        self.channels = ['pose_image']
        self.required_fields = ['process_id', 'image_id', 'target_size']
        logging.info(f"PoseNotificationManager listening on channels: {self.channels}")

    def process_item(self, pose_request):
        """Process a single pose request."""
        return self.process_pose({
            'process_id': pose_request.process_id,
            'image_id': pose_request.image_id,
            'face': pose_request.parameters.get('face', True)
        })

    def process_pose(self, pose_data):
        """Process a pose request through its stages."""
        logging.info(f"Processing pose request: {pose_data}")
        
        try:
            validated_data = self.validate_process_data(pose_data)
            process_id = validated_data['process_id']
            
            def execute_pose_process(session):
                # Get the image from database
                image = session.query(Image).get(validated_data['image_id'])
                if not image:
                    raise ValueError(f"Image {validated_data['image_id']} not found")

                # Get image data
                image_data = image.get_image_data(session)
                if not image_data:
                    raise ValueError("No image data found")

                # Process the pose using detect_pose function
                self.pose_variant = detect_pose(
                    image_data,
                    face=validated_data.get('face', True)
                )
                
                if not self.pose_variant:
                    raise ValueError("Failed to create pose variant")
                    
                return self.pose_variant.id
            
            return self.process_with_error_handling(process_id, execute_pose_process)
            
        except Exception as e:
            logging.error(f"Error processing pose: {str(e)}", exc_info=True)
            raise

    def handle_notification(self, channel, data):
        """Handle pose notifications."""
        logging.info(f"PoseNotificationManager received: channel={channel}, data={data}")
        
        if channel == 'pose_image':
            return self.process_pose(data)
            
        logging.warning(f"Unexpected channel: {channel}")
        return None
