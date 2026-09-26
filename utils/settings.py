"""Shared business contact settings."""

import os
from dotenv import load_dotenv

load_dotenv()

BUSINESS_EMAIL = os.getenv("BUSINESS_EMAIL", "mehro173@gmail.com")
