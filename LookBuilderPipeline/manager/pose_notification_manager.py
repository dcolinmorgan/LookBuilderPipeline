import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant
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
                    # full_data = {
                    #     'process_id': data['process_id'],
                    #     'image_id': data['image_id'],
                    #     'face': process.parameters.get('face')
                    # }
                    logging.info(f"Processing pose with parameters: {full_data}")
                    return self.process_item()
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"PoseNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, pose):
        """Process a single pose notification."""
        # Validate face parameter first
        if not pose.parameters or 'face' not in pose.parameters:
            raise ValueError("Process is missing required face parameter")
        
        validated_data = {
            'process_id': pose.process_id,
            'image_id': pose.image_id,
            'face': pose.parameters.get('face')
        }
        process_id = validated_data['process_id']
        
        def execute_pose_process(session):
            # Get the image
            image = session.query(Image).get(validated_data['image_id'])
            if not image:
                error_msg = (
                    f"Image {validated_data['image_id']} not found in database. "
                    f"Please ensure the image was properly uploaded and exists in the database."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None

            # Check if image has data
            image_data = image.get_image_data(session)
            if not image_data:
                error_msg = (
                    f"Image {validated_data['image_id']} exists but has no data. "
                    f"This could be due to an incomplete upload or data corruption. "
                    f"Try re-uploading the image."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None

            try:
                # Create a temporary ImageVariant instance to use get_or_create_variant, which handles variant-specific logic. 
                # Placing this in the Image class would require importing ImageVariant, leading to circular imports.
                # This approach maintains separation of concerns, keeps logic in the appropriate class, 
                # and provides a clean interface for creating specialized variants using a temporary factory-like object.
                base_variant = ImageVariant(
                    source_image_id=image.image_id,
                    variant_type='pose'
                )
                session.add(base_variant)
                session.flush()
                
                # Now use the source_image_id instead of image_id
                variant = base_variant.get_or_create_variant(
                    session=session,
                    variant_type='pose',
                    face=validated_data['face']
                )
                
                if not variant:
                    error_msg = (
                        f"Failed to create pose variant for image {validated_data['image_id']}. "
                        f"The pose operation completed but returned no variant. "
                        f"This might indicate an issue with the image processing."
                    )
                    logging.error(error_msg)
                    self.mark_process_error(session, process_id, error_msg)
                    return None
                    
                session.expunge(base_variant)
                return variant
                
            except Exception as e:
                error_msg = (
                    f"Error creating pose variant: {str(e)}. "
                    f"This could be due to invalid image data or insufficient system resources."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None
        
        return self.process_with_error_handling(process_id, execute_pose_process)
