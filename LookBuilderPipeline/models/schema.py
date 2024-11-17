"""
This schema defines the database structure for managing image processing tasks.
It includes four tables:

1. **Images**: Stores metadata for all images processed in the system, including original and generated images.
2. **Layers**: Represents the different visual layers associated with images, such as clothing items or backgrounds.
3. **Structures**: Stores structural information related to images, like poses or edge detection data.
4. **ProcessQueue**: Tracks the steps and status of image processing tasks, ensuring the workflow is executed correctly.

Each table is related to the `Images` table through foreign keys, and the `ProcessQueue` table ensures that image processing can proceed step by step.
"""

from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, ForeignKey, Boolean, DateTime, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import OID
from ..config import get_config

# Get the configuration
config = get_config()

# Create an engine for your PostgreSQL database
engine = create_engine(config.SQLALCHEMY_DATABASE_URI)

# Define the Base class
Base = declarative_base()

"""
Images Table:
This table serves as the main repository for images that are uploaded or generated during the image processing workflow.
It stores metadata about each image, including a reference to the binary data, the type of image (original, generated, structure),
and whether the image has been processed or not.

- Each image is uniquely identified by an `image_id`.
- The actual image binary data is stored externally in a large object (LOB) managed by PostgreSQL, and referenced by `image_oid`.
"""
class Images(Base):
    __tablename__ = 'images'
    
    # Unique identifier for each image
    image_id = Column(Integer, primary_key=True)
    
    # Object Identifier (OID) for large object storage, referencing the image data
    image_oid = Column(OID)
    
    # The type of image, indicating whether it's an original, generated, or structure image
    image_type = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime)
    processed = Column(Boolean, default=False, nullable=False)

    def __repr__(self):
        return f"<Image(id={self.image_id}, type={self.image_type}, processed={self.processed})>"

"""
Layers Table:
The Layers table represents different visual layers that are associated with an image.
These layers could include clothing items, pose, backgrounds, or other visual elements that relate to the base image during processing.
Layers are applied separately and associated with a specific image.

- Each layer is uniquely identified by `layer_id`.
- Layers are linked to an image through `image_id`, referencing the Images table.
- The type of layer is defined in `layer_type` (e.g., 'outfit', 'background').
"""
class Layers(Base):
    __tablename__ = 'layers'
    
    # Unique identifier for each layer
    layer_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key referencing the image to which this layer belongs
    image_id = Column(Integer, ForeignKey('images.image_id'), nullable=False)
    
    # The type of layer (e.g., outfit, background, accessories)
    layer_type = Column(String(50), nullable=False)
    
    # Timestamp for when the layer was added to the system
    created_at = Column(TIMESTAMP, default='NOW()')   

"""
Structures Table:
The Structures table stores structural information that describes the composition of an image.
This can include poses, canny edge detection data, or other data that defines the form and layout of the image.
This is particularly useful for pose-based image generation or other structural manipulations.

- Each structure is uniquely identified by `structure_id`.
- Structures are linked to an image through `image_id`, referencing the Images table.
- The type of structure is defined in `structure_type` (e.g., 'pose', 'canny').
"""
class Structures(Base):
    __tablename__ = 'structures'
    
    # Unique identifier for each structure
    structure_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key referencing the image associated with this structure
    image_id = Column(Integer, ForeignKey('images.image_id'), nullable=False)
    
    # The type of structure (e.g., pose, canny for edge detection)
    structure_type = Column(String(50), nullable=False)
    
    # Timestamp for when the structure was added to the system
    created_at = Column(TIMESTAMP, default='NOW()')

"""
ProcessQueue Table:
The ProcessQueue table is designed to track the processing steps for each image in the system.
It defines what the next processing step is for an image and keeps track of the current status of the process.
This table ensures that image processing workflows can be managed and monitored effectively.

- Each process step is uniquely identified by `process_id`.
- Processes are linked to an image through `image_id`, referencing the Images table.
- The `next_step` field defines the next action to be taken in the processing pipeline (e.g., 'retouch', 'layering').
- The `status` field tracks whether the process is 'pending', 'in_progress', or 'completed'.
"""
class ProcessQueue(Base):
    __tablename__ = 'process_queue'
    
    # Unique identifier for each processing step
    process_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key referencing the image being processed
    image_id = Column(Integer, ForeignKey('images.image_id'), nullable=False)
    
    # The next step in the image processing workflow (e.g., 'retouch', 'layering')
    next_step = Column(String(50), nullable=False)
    
    # The current status of the process (e.g., 'pending', 'in_progress', 'completed')
    status = Column(String(50), nullable=False)
    
    # Timestamp for when the process step was added
    created_at = Column(TIMESTAMP, default='NOW()')
    
    # Timestamp for the last update to the process step
    updated_at = Column(TIMESTAMP)

# Create the tables
Base.metadata.create_all(engine)

print("Tables created successfully.")

