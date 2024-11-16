from sqlalchemy import Column, Integer, String, DateTime, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class ProcessQueue(Base):
    __tablename__ = 'process_queue'

    process_id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False)
    next_step = Column(String, nullable=False)
    status = Column(String, nullable=False)
    parameters = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    requested_by = Column(String, nullable=True)  # Client that requested the process
    served_by = Column(String, nullable=True)     # Server/worker that processed it

    VALID_STATUSES = ('pending', 'processing', 'completed', 'error')

    __table_args__ = (
        CheckConstraint(
            status.in_(VALID_STATUSES),
            name='valid_status_check'
        ),
    )