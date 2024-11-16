import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue

from PIL import Image
from io import BytesIO
from LookBuilderPipeline.models.image_variant import ImageVariant
from LookBuilderPipeline.models.image import Image
import select

class LoadPipeNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing LoadPipeNotificationManager")
        self.channels = ['loadpipe']
        logging.info(f"LoadPipeNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id', 'pipe']

    def handle_notification(self, channel, data):
        """Handle loadpipe notifications."""
        logging.info(f"LoadPipeNotificationManager received: channel={channel}, data={data}")
        if channel == 'loadpipe':
            # Get the full process data from ProcessQueue
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Merge the notification data with process parameters
                    full_data = {
                        'process_id': data['process_id'],
                        'image_id': data['image_id'],
                        'pipe': process.parameters.get('pipe'),
                    }
                    logging.info(f"Processing loadpipe with parameters: {full_data}")
                    # if process.parameters.get('pipe') == 'sdxl':
                    #     from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
                    #     self.ImageModelSDXL.prepare_model()
                    #     self.ImageModelSDXL.prepare_image(full_data.keys(['image_path','pose_path','mask_path']))
                    # elif process.parameters.get('pipe') == 'flux':
                    #     from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux
                    #     self.ImageModelFlux.prepare_model()
                    #     self.ImageModelFlux.prepare_image(full_data.keys(['image_path','pose_path','mask_path']))
                    
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"LoadPipeNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, loadpipe):
        """Process a single loadpipe notification."""
        return self.process_loadpipe({
            'process_id': loadpipe.process_id,
            'image_id': loadpipe.image_id,
            'pipe': loadpipe.parameters.get('pipe'),
        })

    def process_loadpipe(self, loadpipe_data):
        """Process a loadpipe notification through its stages."""
        validated_data = self.validate_process_data(loadpipe_data)
        process_id = validated_data['process_id']
        
        def execute_loadpipe_process(session):
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
                variant = image.load_pipe_variant(
                    session,
                    model=validated_data['pipe']
                )
                
                if not variant:
                    error_msg = (
                        f"Failed to create loadpipe variant for image {validated_data['image_id']}. "
                        f"The loadpipe operation completed but returned no variant. "
                        f"This might indicate an issue with the image processing."
                    )
                    logging.error(error_msg)
                    self.mark_process_error(session, process_id, error_msg)
                    return None
                    
                return variant.id
                
            except Exception as e:
                error_msg = (
                    f"Error creating loadpipe variant: {str(e)}. "
                    f"This could be due to invalid image data or insufficient system resources."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None
        
        return self.process_with_error_handling(process_id, execute_loadpipe_process)

    def mark_process_error(self, session, process_id, error_message):
        """Mark a process as error with an error message."""
        process = session.query(ProcessQueue).get(process_id)
        if process:
            process.status = 'error'
            process.error_message = error_message
            session.commit()

    def _listen_for_notifications(self):
        """Override parent method to add more logging"""
        logging.info("Starting loadpipe notification listener thread")
        
        while self.should_listen:
            try:
                if select.select([self.conn], [], [], 1.0)[0]:
                    self.conn.poll()
                    while self.conn.notifies:
                        notify = self.conn.notifies.pop()
                        logging.info(f"loadpipe raw notification received: {notify}")
                        self.notification_queue.put((notify.channel, notify.payload))
                        logging.info(f"loadpipe notification added to queue: {notify.channel}")
            except Exception as e:
                logging.error(f"Error in loadpipe notification listener: {str(e)}")
                self._handle_connection_error()
