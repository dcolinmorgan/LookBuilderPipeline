import threading
from queue import Queue, Empty
import select
import psycopg2
import time
import logging
import json
import os
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError, DBAPIError, DisconnectionError
from psycopg2.errors import OperationalError
import socket
from contextlib import contextmanager

from LookBuilderPipeline.manager.db_manager import DBManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image


class NotificationManager:
    """Core notification management functionality."""
    def __init__(self):
        self.db_manager = DBManager()
        self.session = self.db_manager.get_session()
        self.notification_queue = Queue()
        self.should_listen = True
        self.processor_thread = None
        self.listener_thread = None
        self.conn = None
        self.cursor = None
        self.channels = []  # Subclasses will define their channels
        self.server_name = socket.gethostname()  # Identify this server
        self.client_name = None  # For testing or specific client instances
        self.required_fields = []  # Subclasses will define required fields
        self.process_delay = 0.1  # Time to yield between processing items
        self.max_processing_attempts = 10  # Maximum number of processing attempts
        self.max_attempts = 100  # Safety limit for continuous processing
    
    def setup(self):
        """Set up the notification manager."""
        logging.info("Setting up NotificationManager")
        
        try:
            # Process any existing notifications first
            self.process_existing_queue()
            
            # Set up notification listening
            self.conn = self.db_manager.get_connection()
            self.cursor = self.conn.cursor()
            
            # Listen for notifications on all channels
            for channel in self.channels:
                self.cursor.execute(f"LISTEN {channel};")
                logging.info(f"Listening on channel: {channel}")
            
            self._start_threads()
            logging.info("NotificationManager setup complete")
            
        except Exception as e:
            logging.error(f"Error during NotificationManager setup: {str(e)}")
            raise
    
    def _start_threads(self):
        """Start the listener and processor threads."""
        logging.info("Starting listener and processor threads")
        self.listener_thread = threading.Thread(
            target=self._listen_for_notifications,
            name="NotificationListener"
        )
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
        self.processor_thread = threading.Thread(
            target=self._process_notifications,
            name="NotificationProcessor"
        )
        self.processor_thread.daemon = True
        self.processor_thread.start()

    def _listen_for_notifications(self):
        """Background thread to listen for notifications."""
        logging.info("Starting notification listener thread")
        
        while self.should_listen:
            try:
                if select.select([self.conn], [], [], 1.0)[0]:
                    self.conn.poll()
                    while self.conn.notifies:
                        notify = self.conn.notifies.pop()
                        logging.info(f"Raw notification received: {notify}")
                        self.notification_queue.put((notify.channel, notify.payload))
                        logging.info("Notification added to queue")
            except Exception as e:
                logging.error(f"Error in notification listener: {str(e)}")
                self._handle_connection_error()

    def _handle_connection_error(self):
        """Handle connection errors and attempt reconnection."""
        try:
            logging.info("Attempting to reconnect...")
            self.conn = self.db_manager.get_connection()
            self.cursor = self.conn.cursor()
            for channel in self.channels:
                self.cursor.execute(f"LISTEN {channel};")
            logging.info("Reconnected successfully")
        except Exception as reconnect_error:
            logging.error(f"Reconnection failed: {str(reconnect_error)}")
            time.sleep(5)

    def _process_notifications(self):
        """Background thread to process notifications."""
        logging.info(f"Processor thread {threading.current_thread().name} started")
        
        while self.should_listen:
            try:
                channel, payload = self.notification_queue.get(timeout=1.0)
                logging.info(f"Processing notification: Channel={channel}, Payload={payload}")
                self.process_notification(channel, payload)
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing notification: {str(e)}")
                logging.error("Full traceback:", exc_info=True)

    def process_notification(self, channel, payload):
        """Process a notification."""
        if channel not in self.channels:
            logging.warning(f"Received notification for unsupported channel: {channel}")
            return
            
        try:
            parsed_data = self.parse_payload(payload)
            # Only validate process_id is present
            self.validate_process_data(parsed_data, required_fields=['process_id'])
            
            # Get full process data
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(parsed_data['process_id'])
                if not process:
                    raise ValueError(f"Process {parsed_data['process_id']} not found")
                
                # Create full data dictionary from process
                full_data = {
                    'process_id': process.process_id,
                    'image_id': process.image_id,
                    'next_step': process.next_step,
                    'status': process.status,
                    'parameters': process.parameters or {}
                }
                
                return self.handle_notification(channel, full_data)
        except Exception as e:
            logging.error(f"Error processing notification on channel {channel}: {str(e)}")
            raise

    def handle_notification(self, channel, data):
        """Handle notifications with proper error recovery."""
        logging.info(f"Received notification: channel={channel}, data={data}")
        
        if channel in self.channels:
            try:
                process = self.db_manager.get_process(data['process_id'])
                if process and process.parameters:
                    with self.db_manager.get_session() as session:
                        return self.process_item(process, session)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            except Exception as e:
                logging.error(f"Error handling notification: {str(e)}")
            finally:
                # Ensure connection pool is clean
                self.db_manager.engine.dispose()
        return None

    def process_existing_queue(self):
        """Process one item at a time, checking for more after each completion."""
        try:
            logging.info(f"Checking for existing unprocessed {self.channels} notifications...")
            
            processed_count = 0
            attempts = 0

            while attempts < self.max_attempts:
                attempts += 1
                
                # Get and process one item
                with self.get_managed_session() as session:
                    pending_item = self.get_next_pending_process(session)
                    
                    if not pending_item:
                        if processed_count == 0:
                            logging.info(f"No pending {self.channels} notifications found")
                        else:
                            logging.info(f"Completed processing {processed_count} notifications")
                        return
                    
                    try:
                        self.process_item(pending_item)
                        processed_count += 1
                        logging.info(f"Processed {pending_item.next_step} {pending_item.process_id}")
                    except Exception as e:
                        logging.error(f"Error processing {pending_item.next_step} {pending_item.process_id}: {str(e)}")

            if attempts >= self.max_attempts:
                logging.warning(f"Reached maximum processing attempts ({self.max_attempts})")

        except Exception as e:
            logging.error(f"Error in queue processing: {str(e)}")
            raise

    def process_item(self, item):
        """Process a single queue item. To be implemented by subclasses."""
        raise NotImplementedError

    def stop(self):
        """Stop the notification manager."""
        logging.info("Stopping NotificationManager...")
        self.should_listen = False
        
        for thread in [self.listener_thread, self.processor_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
                if thread.is_alive():
                    logging.warning(f"{thread.name} did not stop cleanly")
        
        for conn in [self.cursor, self.conn, self.session]:
            if conn:
                conn.close()
            
        logging.info("NotificationManager stopped")

    def create_process(self, image_id: int, next_step: str, status: str = 'pending', parameters: dict = None) -> ProcessQueue:
        """Create a new process in the queue."""
        return ProcessQueue(
            image_id=image_id,
            next_step=next_step,
            status=status,
            parameters=parameters or {},
            requested_by=f"LookBuilderPipeline@{self.server_name}",  # Updated format
        )

    def get_processes_for_client(self, session, **filters):
        """Get processes with client filtering."""
        query = session.query(ProcessQueue)
        
        # Add any filters passed in
        for key, value in filters.items():
            query = query.filter_by(**{key: value})
            
        # Add client filter if set
        if self.client_name:
            query = query.filter_by(requested_by=self.client_name)
            
        return query

    def update_process_status(self, session, process_id, status):
        """Update process status with server tracking."""
        process = session.query(ProcessQueue)\
            .filter(ProcessQueue.process_id == process_id)\
            .with_for_update()\
            .first()
        
        if not process:
            raise ValueError(f"Process {process_id} not found")
            
        process.status = status
        process.updated_at = datetime.now()
        process.served_by = f"LookBuilderPipeline@{self.server_name}"
        return process

    def parse_payload(self, payload):
        """Parse JSON payload with error handling."""
        try:
            if isinstance(payload, str):
                return json.loads(payload)
            return payload
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON payload: {str(e)}")
            raise ValueError(f"Invalid JSON payload: {str(e)}")

    def validate_process_data(self, data, required_fields=None):
        """Validate process data has required fields."""
        fields_to_check = required_fields or ['process_id']  # Default to only requiring process_id
        missing_fields = [field for field in fields_to_check if not data.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        return data

    @contextmanager
    def get_managed_session(self):
        """Context manager for database sessions with error handling."""
        with self.db_manager.get_session() as session:
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                logging.error(f"Database session error: {str(e)}")
                raise

    def handle_process_error(self, process_id, error, status='error'):
        """Common error handling for process updates."""
        logging.error(f"Process {process_id} error: {str(error)}")
        try:
            with self.get_managed_session() as session:
                self.update_process_status(session, process_id, status)
        except Exception as e:
            logging.error(f"Failed to update error status for process {process_id}: {str(e)}")
        raise error

    def process_with_error_handling(self, process_id, processing_func):
        """Execute a process function with standard error handling."""
        try:
            with self.get_managed_session() as session:
                # Update to processing
                self.update_process_status(session, process_id, 'processing')
                
                # Execute the process
                result = processing_func(session)
                
                # Update to completed
                self.update_process_status(session, process_id, 'completed')
                 
                return result
        except Exception as e:
            return self.handle_process_error(process_id, e)

    def get_next_pending_process(self, session):
        """Get a single pending process."""
        return self.get_processes_for_client(
            session,
            next_step=self.channels[0],
            status='pending'
        ).first()

    def get_image(self, image_id):
        """Retrieve image data from database/storage."""
        with self.get_managed_session() as session:
            image = session.query(Image)\
                .filter_by(image_id=image_id)\
                .first()
            if not image:
                raise ValueError(f"Image {image_id} not found")
            return image.data  # Assuming your Image model has a 'data' field

        
    def mark_process_error(self, session, process_id, error_message):
        """Mark a process as error with an error message."""
        process = session.query(ProcessQueue).get(process_id)
        if process:
            process.status = 'error'
            process.error_message = error_message
            session.commit()
