from sqlalchemy import Column, Integer, String, DateTime, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class ProcessQueue(Base):
    __tablename__ = 'process_queue'

    process_id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False)
    next_step = Column(String(50), nullable=False)
    parameters = Column(JSONB, nullable=True, server_default='{}')
    status = Column(String(20), nullable=False, default='pending')
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now)

    VALID_STATUSES = ('pending', 'processing', 'completed', 'error')

    __table_args__ = (
        CheckConstraint(
            status.in_(VALID_STATUSES),
            name='valid_status_check'
        ),
    )