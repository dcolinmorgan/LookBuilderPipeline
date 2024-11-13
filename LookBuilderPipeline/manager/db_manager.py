import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from typing import Optional
from ..models.user import User

class DBManager:
    def __init__(self):
        db_url = f"postgresql://{os.getenv('DB_USERNAME', 'lookbuilderhub_user')}:" \
                 f"{os.getenv('DB_PASSWORD', 'lookbuilderhub_password')}@" \
                 f"{os.getenv('DB_HOST', 'localhost')}/" \
                 f"{os.getenv('DB_NAME', 'lookbuilderhub_db')}"
        
        self.db_url = db_url
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        """Get a SQLAlchemy session."""
        return self.Session()

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
        
        Args:
            session: SQLAlchemy session
            user_id (Optional[int]): User ID to look up
            email (Optional[str]): User email to look up
            
        Returns:
            Optional[User]: User object if found, None otherwise
            
        Raises:
            ValueError: If neither user_id nor email is provided
        """
        if user_id is None and email is None:
            raise ValueError("Must provide either user_id or email")
            
        query = session.query(User)
        
        if user_id is not None:
            return query.get(user_id)
            
        if email is not None:
            return query.filter(User.email == email).first()
            
        return None
    
    
        