import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("database.connection")

# Load configuration from .env file
load_dotenv()

# Retrieve MySQL settings from environment
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "voice_agent")

logger.info(f"Loaded database configuration. Host: {MYSQL_HOST}, Port: {MYSQL_PORT}, Database: {MYSQL_DATABASE}, User: {MYSQL_USER}")

# Connect via PyMySQL
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# Create SQLAlchemy engine with connection pooling settings
logger.info("Initializing SQLAlchemy engine...")
engine = create_engine(
    DATABASE_URL,
    pool_recycle=3600,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    logger.info("Creating a new database session LocalSession...")
    db = SessionLocal()
    try:
        yield db
    finally:
        logger.info("Closing database session LocalSession.")
        db.close()

if __name__ == "__main__":
    logger.info("Testing database connection directly...")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Connection test query 'SELECT 1' executed successfully.")
        print("\n Success: Database connection established successfully!")
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        print("\n❌ Error: Failed to connect to the database!")

