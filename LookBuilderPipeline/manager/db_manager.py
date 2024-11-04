from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, Boolean, func
import psycopg2

from sqlalchemy.dialects.postgresql import OID

from LookBuilderPipeline.config import Config
class DBManager:
    def __init__(self, db_url):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        self._session = None  # Initialize _session to None
        