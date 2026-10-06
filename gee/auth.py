"""Earth Engine initialisation from a service account or existing user credentials, failing fast when absent."""
from __future__ import annotations

import os
from collections.abc import Mapping

import ee

SERVICE_ACCOUNT_ENV = "EE_SERVICE_ACCOUNT"
KEY_FILE_ENV = "EE_KEY_FILE"
PROJECT_ENV = "EE_PROJECT"
BLOCKED_MESSAGE = (
    "BLOCKED: no Earth Engine credentials. Set EE_SERVICE_ACCOUNT and EE_KEY_FILE (path to the service-account "
    "JSON key), or run `earthengine authenticate` to create user credentials; set EE_PROJECT to the Cloud project.")


class EarthEngineUnavailable(RuntimeError):
    """Raised when Earth Engine cannot be used (missing credentials or failed initialisation)."""


def user_credentials_path() -> str:
    """Path where `earthengine authenticate` stores user credentials."""
    return ee.oauth.get_credentials_path()


def credential_source(env: Mapping[str, str]) -> str | None:
    """Return 'service_account', 'user' or None depending on which credentials are available."""
    account, key_file = env.get(SERVICE_ACCOUNT_ENV), env.get(KEY_FILE_ENV)
    if account and key_file and os.path.isfile(key_file):
        return "service_account"
    if os.path.isfile(user_credentials_path()):
        return "user"
    return None


def initialize(env: Mapping[str, str] | None = None) -> str:
    """Initialise Earth Engine and return the credential source; raise EarthEngineUnavailable otherwise."""
    env = os.environ if env is None else env
    source = credential_source(env)
    if source is None:
        raise EarthEngineUnavailable(BLOCKED_MESSAGE)
    project = env.get(PROJECT_ENV) or None
    try:
        if source == "service_account":
            credentials = ee.ServiceAccountCredentials(env[SERVICE_ACCOUNT_ENV], env[KEY_FILE_ENV])
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project)
    except Exception as error:
        raise EarthEngineUnavailable(f"BLOCKED: Earth Engine initialisation failed ({source}): {error}") from error
    return source
