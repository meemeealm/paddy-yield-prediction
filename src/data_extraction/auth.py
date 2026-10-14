"""
Shared Earth Engine authentication helper.

Call `initialize_gee()` once at the start of any script; no manual
`ee.Initialize()` or login step is needed in the scripts themselves.

Resolution order:
  1. Service account: if GOOGLE_APPLICATION_CREDENTIALS points to a service
     account JSON key, it is used (no browser login, works on servers).
  2. Personal account: otherwise falls back to your saved Earth Engine
     credentials. If none exist yet, a one-time browser login is started
     automatically and the credentials are saved for next time.

Settings can live in a `.env` file next to the scripts (loaded automatically
if python-dotenv is installed):

    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
    GEE_PROJECT_NAME=your-gee-project
"""

import json
import logging
import os

import ee

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

logger = logging.getLogger(__name__)


def initialize_gee() -> None:
    """Initialize Earth Engine automatically (service account or saved personal credentials)."""
    project = os.getenv("GEE_PROJECT_NAME")
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # 1. Service account
    if key_path:
        with open(key_path) as fh:
            service_account = json.load(fh)["client_email"]
        credentials = ee.ServiceAccountCredentials(service_account, key_path)
        ee.Initialize(credentials, project=project)
        logger.info("Earth Engine initialized with service account %s", service_account)
        return

    # 2. Personal account (saved credentials, with one-time login if missing)
    try:
        ee.Initialize(project=project)
    except Exception:
        logger.info("No saved Earth Engine credentials found -- starting one-time authentication ...")
        ee.Authenticate()
        ee.Initialize(project=project)
    logger.info("Earth Engine initialized with personal credentials")
