from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql import func
from typing import List
from werkzeug.security import check_password_hash

from LookBuilderPipeline.models.image import Image

from .base import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String)
    password_hash = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    # Relationships
    images: Mapped[List["Image"]] = relationship("Image", back_populates="user")
    looks = relationship("Look", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_user_email', 'email'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
        
    @classmethod
    def get_by_email(cls, email):
        """Get a user by their email"""
        from LookBuilderPipeline.manager.db_manager import DBManager
        
        db_manager = DBManager()
        with db_manager.get_session() as session:
            try:
                return db_manager.get_user(session, email=email)
            except Exception as e:
                logging.error(f"Error in get_by_email: {str(e)}")
                raise


    def check_password(self, password: str) -> bool:
        """
        Check if the provided password matches the stored hash.
        
        Args:
            password (str): The password to check
            
        Returns:
            bool: True if password matches, False otherwise
        """
        return check_password_hash(self.password_hash, password)
