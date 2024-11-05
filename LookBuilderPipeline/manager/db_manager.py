import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class DBManager:
    def __init__(self):
        db_url = f"postgresql://{os.getenv('DB_USERNAME', 'lookbuilderhub_user')}:" \
                 f"{os.getenv('DB_PASSWORD', 'lookbuilderhub_password')}@" \
                 f"{os.getenv('DB_HOST', 'localhost')}/" \
                 f"{os.getenv('DB_NAME', 'lookbuilderhub_db')}"
        
        self.engine = create_engine(
            db_url,
            isolation_level="READ COMMITTED",  # Explicit isolation level
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )
        
        self.Session = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
    
    def get_session(self):
        return self.Session()

    def get_engine(self):
        """Get the SQLAlchemy engine."""
        return self.engine
        