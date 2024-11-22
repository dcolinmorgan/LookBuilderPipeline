import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
import select

class SDXLNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing SDXLNotificationManager")
        self.channels = ['image_sdxl']
        logging.info(f"SDXLNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id','prompt']

    def handle_notification(self, channel, data):
        """Handle sdxl notifications."""
        logging.info(f"SDXLNotificationManager received: channel={channel}, data={data}")
        if channel == 'image_sdxl':
            # Get the full process data from ProcessQueue
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Merge the notification data with process parameters
                    full_data = {
                        'process_id': data['process_id'],
                        'image_id': data['image_id'],
                        # 'image_pose_id': data['image_pose_id'],
                        # 'image_mask_id': data['image_mask_id'],
                        # 'prompt': 'purple haired woman ice skating in winter park', #process.parameters.get('prompt'),
                        # 'negative_prompt': 'ugly, deformed',#process.parameters.get('negative_prompt')
                    }
                    logging.info(f"Processing sdxl with parameters: {full_data}")
                    return self.process_sdxl(full_data)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"SDXLNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, sdxl):
        """Process a single sdxl notification."""
        return self.process_sdxl({
            'process_id': sdxl.process_id,
            'image_id': sdxl.image_id,
            # 'image_pose_id': sdxl.image_pose_id,
            # 'image_mask_id': sdxl.image_mask_id,
            # 'prompt': sdxl.prompt,
            # 'negative_prompt': sdxl.negative_prompt
        })

    def process_sdxl(self, sdxl_data):
        """Process a sdxl notification through its stages."""
        validated_data = self.validate_process_data(sdxl_data)
        # if 'image_pose_id' not in validated_data:
        #     logging.error("image_pose_id is missing from validated_data.")
        #     return None  # Handle the error as appropriate

        process_id = validated_data['process_id']
        
        def execute_sdxl_process(session):
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
                self.session = session
                variant = image.get_or_create_sdxl_variant(
                    session=self.session,
                    # image_pose_id=validated_data['image_pose_id'],
                    # image_mask_id=validated_data['image_mask_id'],
                    prompt='purple haired woman ice skating in winter park', #validated_data['prompt'],
                    negative_prompt='ugly, deformed',#validated_data['negative_prompt']
                )
                
                if not variant:
                    error_msg = (
                        f"Failed to create sdxl variant for image {validated_data['image_id']}. "
                        f"The sdxl operation completed but returned no variant. "
                        f"This might indicate an issue with the image processing."
                    )
                    logging.error(error_msg)
                    self.mark_process_error(session, process_id, error_msg)
                    return None
                    
                return variant
                
            except Exception as e:
                error_msg = (
                    f"Error creating sdxl variant: {str(e)}. "
                    f"This could be due to invalid image data or insufficient system resources."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None
        
        return self.process_with_error_handling(process_id, execute_sdxl_process)

    def mark_process_error(self, session, process_id, error_message):
        """Mark a process as error with an error message."""
        process = session.query(ProcessQueue).get(process_id)
        if process:
            process.status = 'error'
            process.error_message = error_message
            session.commit()

    def _listen_for_notifications(self):
        """Override parent method to add more logging"""
        logging.info("Starting sdxl notification listener thread")
        
        while self.should_listen:
            try:
                if select.select([self.conn], [], [], 1.0)[0]:
                    self.conn.poll()
                    while self.conn.notifies:
                        notify = self.conn.notifies.pop()
                        logging.info(f"sdxl raw notification received: {notify}")
                        self.notification_queue.put((notify.channel, notify.payload))
                        logging.info(f"sdxl notification added to queue: {notify.channel}")
            except Exception as e:
                logging.error(f"Error in sdxl notification listener: {str(e)}")
                self._handle_connection_error()
