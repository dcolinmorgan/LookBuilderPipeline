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
from LookBuilderPipeline.models.proccess_queue import ProcessQueue


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
        """Process a notification. To be implemented by subclasses."""
        raise NotImplementedError

    def process_existing_queue(self):
        """Process existing queue. To be implemented by subclasses."""
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

    def create_process(self, **kwargs):
        """Create a new process with server identification."""
        process = ProcessQueue(
            requested_by=self.client_name,
            **kwargs
        )
        return process

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
        process.served_by = self.server_name
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
        fields_to_check = required_fields or self.required_fields
        missing_fields = [field for field in fields_to_check if not data.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        return data

    @contextmanager
    def get_managed_session(self):
        """Context manager for database sessions with error handling."""
        session = self.db_manager.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

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

