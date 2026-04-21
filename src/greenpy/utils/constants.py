import os
from dotenv import load_dotenv

load_dotenv()

JAVA_HOME = os.getenv("JAVA_HOME", "")
GEE_PROJECT_NAME = os.getenv("GEE_PROJECT_NAME", "")
