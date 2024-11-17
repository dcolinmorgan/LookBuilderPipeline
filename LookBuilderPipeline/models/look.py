from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from LookBuilderPipeline.models.base import Base

class Look(Base):
    __tablename__ = 'looks'

    look_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    user_id = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)

    def __repr__(self):
        return f"<Look(look_id={self.look_id}, name='{self.name}', user_id={self.user_id})>" 