import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging
from typing import Optional
from ..models.user import User
from ..config import get_config

class DBManager:
    def __init__(self):
        config = get_config()()  # Get the config instance
        
        # Get database URL from config
        self.db_url = config.SQLALCHEMY_DATABASE_URI
        logging.info(f"Connecting to database at {config.DB_HOST}")
        
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

        # Initialize other attributes
        self.channels = []
        self.should_listen = True
        self.conn = None
        self.notification_queue = None

    @contextmanager
    def get_session(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    def safe_commit(self, session):
        """Safely commit the session and handle any errors."""
        try:
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Error committing to database: {str(e)}")
            return False

    def get_connection(self):
        """Get a raw psycopg2 connection for LISTEN/NOTIFY."""
        config = get_config()()  # Get the config instance
        conn = psycopg2.connect(config.PSYCOPG2_CONNECTION_STRING)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn

    def get_user(self, user_id: Optional[int] = None, email: Optional[str] = None) -> Optional[User]:
        """Get a user by ID or email."""
        if user_id is None and email is None:
            raise ValueError("Must provide either user_id or email")
        
        with self.get_session() as session:
            try:
                query = session.query(User)
                if user_id is not None:
                    return query.get(user_id)
                return query.filter(User.email == email).first()
            except Exception as e:
                logging.error(f"Error getting user: {str(e)}")
                raise

    def listen(self, channel):
        """Listen to a specific notification channel."""
        self.channels.append(channel)
        if self.conn is None:
            self.conn = self.get_connection()
        cur = self.conn.cursor()
        cur.execute(f"LISTEN {channel};")
        logging.info(f"Listening on channel: {channel}")

    def unlisten(self, channel):
        """Stop listening to a specific channel."""
        if channel in self.channels:
            self.channels.remove(channel)
        if self.conn:
            cur = self.conn.cursor()
            cur.execute(f"UNLISTEN {channel};")
            logging.info(f"Stopped listening on channel: {channel}")

    def notify(self, channel, payload):
        """Send a notification on a specific channel."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"NOTIFY {channel}, %s;", (payload,))
            logging.info(f"Notification sent on channel {channel}: {payload}")

    def handle_notifications(self):
        """Handle incoming notifications."""
        if self.conn is None:
            self.conn = self.get_connection()
        
        while self.should_listen:
            if self.conn.notifies:
                notify = self.conn.notifies.pop()
                logging.info(f"Received notification: {notify.payload}")
                if self.notification_queue:
                    self.notification_queue.put(notify)

        