from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import OID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from PIL import Image as PILImage
import io
from typing import Optional
import logging
from datetime import datetime
from .base import Base

class Image(Base):
    __tablename__ = 'images'

    image_id = Column(Integer, primary_key=True)
    image_oid = Column(Integer)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.now)

    image_type = Column(String(10), nullable=False)  
    updated_at = Column(DateTime) 
    processed = Column(Boolean, default=False)  

    # Use string reference for User
    user = relationship("User", back_populates="images")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_db_manager(cls):
        from LookBuilderPipeline.manager.db_manager import DBManager
        return DBManager()

    @classmethod
    def get_by_id(cls, image_id: int):
        """Get an image by its ID"""
        db_manager = cls.get_db_manager()
        with db_manager.get_session() as session:
            image = session.query(cls).get(image_id)
            if image:
                session.expunge(image)
            return image

    def save(self):
        """Save the image to the database"""
        db_manager = self.get_db_manager()
        with db_manager.get_session() as session:
            session.add(self)
            session.flush()
            image_id = self.image_id
            session.expunge(self)
            return image_id

    def update(self, **kwargs):
        """Update image attributes"""
        db_manager = self.get_db_manager()
        with db_manager.get_session() as session:
            session.add(self)
            for key, value in kwargs.items():
                setattr(self, key, value)
            session.flush()
            session.expunge(self)


    def get_image_data(self, session):
        """Get the image data from the large object storage."""
        logging.info(f"Attempting to get image data for image_id={self.image_id}, image_oid={self.image_oid}")
        
        if not self.image_oid:
            logging.error(f"No image_oid found for image {self.image_id}")
            return None
            
        try:
            logging.info(f"Creating lobject for image {self.image_id} with oid {self.image_oid}")
            connection = session.connection().connection
            
            lob = connection.lobject(oid=self.image_oid, mode='rb')
            logging.info(f"Successfully created lobject for image {self.image_id}")
            
            data = lob.read()
            logging.info(f"Successfully read {len(data)} bytes from image {self.image_id}")
            
            lob.close()
            return data
            
        except Exception as e:
            logging.error(f"Error reading image data for image {self.image_id}: {str(e)}", exc_info=True)
            return None

