from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ProcessQueue(Base):
    __tablename__ = 'process_queue'

    process_id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False, index=True)
    next_step = Column(String(50), nullable=False)
    parameters = Column(JSON)
    status = Column(String(20), default='pending')
    created_at = Column(DateTime, server_default='CURRENT_TIMESTAMP')
    updated_at = Column(DateTime)