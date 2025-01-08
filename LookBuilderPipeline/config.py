import os
from datetime import datetime 

class Config:
    # Base database configuration
    DB_USERNAME = os.environ.get('DB_USERNAME', 'lookbuilderhub_user')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_NAME = os.environ.get('DB_NAME', 'lookbuilderhub_db')
    DB_PORT = os.environ.get('DB_PORT', '5432')

    # Debug logging of database configuration
    print(datetime.now().time())
    print("Environment variables:")
    print(f"DB_USERNAME: {os.getenv('DB_USERNAME')}")
    print(f"DB_HOST: {os.getenv('DB_HOST')}")
    print(f"DB_NAME: {os.getenv('DB_NAME')}")

    # SQLAlchemy database URI
    @property
    def SQLALCHEMY_DATABASE_URI(self):
        if not self.DB_PASSWORD:
            return f'postgresql+psycopg2://{self.DB_USERNAME}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'
        return f'postgresql+psycopg2://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'
    
    # psycopg2 connection string
    @property
    def PSYCOPG2_CONNECTION_STRING(self):
        return f"dbname={self.DB_NAME} user={self.DB_USERNAME} password={self.DB_PASSWORD} host={self.DB_HOST} port={self.DB_PORT}"

    # Other configuration settings
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')

    # Add more configuration variables as needed

class DevelopmentConfig(Config):
    """Local development configuration"""
    DEBUG = True
    DB_USERNAME = 'lookbuilderhub_user'
    DB_PASSWORD = ''
    DB_HOST = 'localhost'
    DB_NAME = 'lookbuilderhub_db'
    DB_PORT = '5432'

class AlphaConfig(Config):
    """Alpha environment configuration"""
    DEBUG = True
    DB_USERNAME = 'lookbuilderhub_user'
    DB_PASSWORD = 'svLjBtkiTOMyYND7MTXJ7EnGBymPo9n4'
    DB_HOST = 'dpg-cspvql5ds78s73dd4020-a'
    DB_NAME = 'lookbuilderhub_db'
    DB_PORT = '5432'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Production database settings would go here

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'alpha': AlphaConfig,
    'production': ProductionConfig,
    'default': AlphaConfig
}

def get_config():
    """Get the active configuration based on FLASK_ENV"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config[env]

