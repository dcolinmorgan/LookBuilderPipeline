import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

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
        