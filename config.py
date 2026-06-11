import streamlit as st

def _s(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

DB_CONFIG = {
    "host":     _s("db_host"),
    "port":     int(_s("db_port", 5432)),
    "dbname":   _s("db_name"),
    "user":     _s("db_user"),
    "password": _s("db_password"),
    "sslmode":  _s("db_sslmode", "require"),
}

BOLT_CLIENT_ID     = _s("bolt_client_id")
BOLT_CLIENT_SECRET = _s("bolt_client_secret")
BOLT_TOKEN_URL     = _s("bolt_token_url", "https://oidc.bolt.eu/token")
BOLT_API_BASE      = _s("bolt_api_base",  "https://api.bolt.eu")

TOP_DRIVER_IDS = [128, 81, 155, 123, 180, 230, 228, 130, 182, 195]
BAD_DRIVER_IDS = [82, 178, 72, 36, 32]

DRIVER_NAMES = {
    128: "Monier Janabi",
    81:  "Marius Norvaisas",
    155: "Ertac Cindogulu",
    123: "Bal Jamts",
    180: "Abdi Saeed Mohamed",
    230: "Anish Chaudhry",
    228: "MHD Amir Aljaghsi",
    130: "Jermaine Asante Gyamfi",
    182: "Mohamed Warsame Nur",
    195: "Brijenkumar Patel",
    82:  "Aaron Bartley",
    178: "Abdullahi Saleh",
    72:  "Ponki Miah",
    36:  "Angeline Lewis",
    32:  "Emran Uddin",
}
