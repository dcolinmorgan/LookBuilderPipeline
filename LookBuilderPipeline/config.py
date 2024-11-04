import os
from datetime import datetime; 

class Config:
    # Database configuration
    DB_USERNAME = os.environ.get('DB_USERNAME', 'default_username')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'default_password')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_NAME = os.environ.get('DB_NAME', 'mydatabase')


   
    print(datetime.now().time())
    print("Environment variables:")
    print(f"DB_USERNAME: {os.getenv('DB_USERNAME')}")
    print(f"DB_HOST: {os.getenv('DB_HOST')}")
    print(f"DB_NAME: {os.getenv('DB_NAME')}")

    print('DB_USERNAME:', DB_USERNAME)
    print('DB_HOST:', DB_HOST)
    print('DB_NAME:', DB_NAME)


    # SQLAlchemy database URI
    SQLALCHEMY_DATABASE_URI = f'postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
    
    # psycopg2 connection string
    PSYCOPG2_CONNECTION_STRING = f"dbname={DB_NAME} user={DB_USERNAME} password={DB_PASSWORD} host={DB_HOST}"

    # Other configuration settings
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')

    # Add more configuration variables as needed

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

# You can add more configuration classes for different environments

# Set the active configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    return config[os.environ.get('FLASK_ENV', 'default')]

