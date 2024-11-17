from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql import func
from typing import List
from werkzeug.security import check_password_hash

from LookBuilderPipeline.models.image import Image

from .base import Base

class User(Base):
    __tablename__ = 'users'

    user_id: Mapped[int] = Column(Integer, primary_key=True)
    email: Mapped[str] = Column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = Column(String(255), nullable=False)
    created_at: Mapped[DateTime] = Column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime] = Column(DateTime, onupdate=func.now())

    # Relationships
    images: Mapped[List["Image"]] = relationship("Image", back_populates="user")
    looks = relationship("Look", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_user_email', 'email'),
    )

    def __repr__(self):
        return f"<User(id={self.user_id}, email={self.email})>"

    def check_password(self, password: str) -> bool:
        """
        Check if the provided password matches the stored hash.
        
        Args:
            password (str): The password to check
            
        Returns:
            bool: True if password matches, False otherwise
        """
        return check_password_hash(self.password_hash, password)
