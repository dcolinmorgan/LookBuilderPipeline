import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from typing import Optional
from ..models.user import User
from contextlib import contextmanager
import logging
from sqlalchemy.orm import joinedload

class DBManager:
    def __init__(self):
        db_url = f"postgresql://{os.getenv('DB_USERNAME', 'lookbuilderhub_user')}:" \
                 f"{os.getenv('DB_PASSWORD', 'lookbuilderhub_password')}@" \
                 f"{os.getenv('DB_HOST', 'localhost')}/" \
                 f"{os.getenv('DB_NAME', 'lookbuilderhub_db')}"
        
        self.db_url = db_url
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
            logging.error(f"Database session error: {str(e)}")
            session.rollback()
            raise
        finally:
            try:
                # Ensure connection is valid
                session.execute('SELECT 1')
                session.close()
            except Exception:
                # If connection is invalid, remove it from pool
                session.bind.dispose()
                session.close()

    def safe_commit(self, session):
        """Safely commit changes with error handling."""
        try:
            session.commit()
            return True
        except Exception as e:
            logging.error(f"Commit error: {str(e)}")
            session.rollback()
            return False

    def get_connection(self):
        """Get a raw psycopg2 connection for LISTEN/NOTIFY."""
        conn = psycopg2.connect(self.db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn

    def get_connection_params(self):
        """Get connection parameters from db_url."""
        return {
            'dbname': os.getenv('DB_NAME', 'lookbuilderhub_db'),
            'user': os.getenv('DB_USERNAME', 'lookbuilderhub_user'),
            'password': os.getenv('DB_PASSWORD', 'lookbuilderhub_password'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
    
    def get_user(self, session, user_id: Optional[int] = None, email: Optional[str] = None) -> Optional[User]:
        """
        Get a user by ID or email.
        Returns a User object.
        """
        if user_id is None and email is None:
            raise ValueError("Must provide either user_id or email")
        
        try:
            query = session.query(User)
            if user_id is not None:
                return query.get(user_id)
            else:
                return query.filter(User.email == email).first()
                
        except Exception as e:
            logging.error(f"Error getting user: {str(e)}")
            raise
    
    
        