from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql import func
from typing import List
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import logging, os
from flask_login import UserMixin
from .base import Base

class User(Base, UserMixin):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    images: Mapped[List["Image"]] = relationship("Image", back_populates="user", lazy="dynamic")
    looks = relationship("Look", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_user_email', 'email'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
        
    @classmethod
    def get_by_email(cls, email: str):
        """Get a user by their email and prepare it for Flask-Login"""
        from LookBuilderPipeline.manager.db_manager import DBManager
        
        db_manager = DBManager()
        with db_manager.get_session() as session:
            user = session.query(cls).filter(cls.email == email).first()
            if user:
                # Load any necessary attributes while session is open
                _ = user.id
                _ = user.email
                _ = user.password_hash
                
                # Detach from session
                session.expunge(user)
                
            return user

    @classmethod
    def get(cls, user_id: int):
        """Get a user by ID and prepare it for Flask-Login"""
        from LookBuilderPipeline.manager.db_manager import DBManager
        
        db_manager = DBManager()
        with db_manager.get_session() as session:
            user = session.query(cls).get(user_id)
            if user:
                # Load any necessary attributes while session is open
                _ = user.id
                _ = user.email
                _ = user.password_hash
                
                # Detach from session
                session.expunge(user)
                
            return user

    @classmethod
    def authenticate(cls, email: str, password: str):
        """Authenticate a user and prepare for Flask-Login"""
        user = cls.get_by_email(email)
        if user and user.check_password(password):
            return user
        return None

    def set_password(self, password: str) -> None:
        """Set the user's password hash"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Check if the provided password matches the hash"""
        if self.password_hash:
            return check_password_hash(self.password_hash, password)
        return False

    # Flask-Login methods
    def get_id(self):
        return str(self.id)

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False
    
    @classmethod
    def create_test_user(cls, db_session):
        """Create a test user if it doesn't exist and we're in dev environment."""
        print(f"Creating test user in {os.environ.get('FLASK_ENV')} environment")
        # Check if we're in a development or alpha environment
        env = os.environ.get('FLASK_ENV', 'development')
        if env in ['development', 'alpha']:
            # Development/testing logic here
            logging.info("Creating test user")    
            test_email = 'test@test.com'
            test_password = 'test123'
            
            try:
                # Check if test user already exists
                existing_user = cls.get_by_email(test_email)
                if existing_user:
                    logging.info(f"Test user {test_email} already exists")
                    return existing_user
                    
                # Create new test user
                logging.info(f"Creating new test user: {test_email}")
                new_user = cls()
                new_user.email = test_email
                new_user.set_password(test_password)
                
                db_session.add(new_user)
                db_session.commit()
                
                logging.info(f"Created test user: {test_email}")
                return new_user
                
            except Exception as e:
                db_session.rollback()
                logging.error(f"Error creating test user: {str(e)}")
                return None
        else:
            # Production logic here
            return None
