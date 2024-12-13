import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant

import select

class OutfitNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing OutfitNotificationManager")
        self.channels = ['image_outfit']
        logging.info(f"OutfitNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id', 'inverse']

    def handle_notification(self, channel, data):
        """Handle outfit notifications."""
        logging.info(f"OutfitNotificationManager received: channel={channel}, data={data}")
        if channel == 'image_outfit':
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
                    logging.info(f"Processing outfit with parameters: {full_data}")
                    return self.process_outfit(full_data)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"OutfitNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, outfit):
        """Process a single outfit notification."""
        return self.process_outfit({
            'process_id': outfit.process_id,
            'image_id': outfit.image_id,
            'inverse': outfit.parameters.get('inverse')
        })

    def process_outfit(self, outfit_data):
        """Process a outfit notification through its stages."""
        validated_data = self.validate_process_data(outfit_data)
        process_id = validated_data['process_id']
        
        def execute_outfit_process(session):
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
                # Create a temporary ImageVariant instance to use get_or_create_variant
                base_variant = ImageVariant(
                    source_image_id=image.image_id,
                    variant_type='outfit'
                )
                session.add(base_variant)
                session.flush()
                
                variant, _ = base_variant.get_or_create_variant(
                    session=session,
                    variant_type='outfit',
                    inverse=validated_data['inverse']
                )
                
                if not variant:
                    error_msg = (
                        f"Failed to create outfit variant for image {validated_data['image_id']}. "
                        f"The outfit operation completed but returned no variant. "
                        f"This might indicate an issue with the image processing."
                    )
                    logging.error(error_msg)
                    self.mark_process_error(session, process_id, error_msg)
                    return None
                    
                return variant
                
            except Exception as e:
                error_msg = (
                    f"Error creating outfit variant: {str(e)}. "
                    f"This could be due to invalid image data or insufficient system resources."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None
        
        return self.process_with_error_handling(process_id, execute_outfit_process)
