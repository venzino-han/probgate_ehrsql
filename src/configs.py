# get config from .env file
from dotenv import load_dotenv
import os
load_dotenv()

# define as values 
DB_ID = os.getenv("DB_ID")
TABLES_PATH = os.getenv("TABLES_PATH")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
TRAIN_LABEL_PATH = os.getenv("TRAIN_LABEL_PATH")
VALID_DATA_PATH = os.getenv("VALID_DATA_PATH")
DB_PATH = os.getenv("DB_PATH")
NEW_TRAIN_DIR = os.getenv("NEW_TRAIN_DIR")
NEW_VALID_DIR = os.getenv("NEW_VALID_DIR")
NEW_TEST_DIR = os.getenv("NEW_TEST_DIR")
RESULT_DIR = os.getenv("RESULT_DIR")