from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql import func
from typing import List

from .base import Base

class User(Base):
    __tablename__ = 'user'

    id: Mapped[int] = Column(Integer, primary_key=True)
    email: Mapped[str] = Column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = Column(String(255), nullable=False)
    created_at: Mapped[DateTime] = Column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime] = Column(DateTime, onupdate=func.now())

    # Relationships
    images: Mapped[List["Image"]] = relationship("Image", back_populates="user")
    image_variants: Mapped[List["ImageVariant"]] = relationship("ImageVariant", back_populates="user")

    __table_args__ = (
        Index('idx_user_email', 'email'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
