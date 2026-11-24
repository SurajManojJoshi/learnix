"""
Configuration management for AI Learning & Study Assistant.
Handles paths, environment variables, Gemini API setup, and model parameters.
"""

import os
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress deprecation warnings from legacy packages
warnings.filterwarnings("ignore", category=FutureWarning)

# Fix for Windows Python 3.8+ DLL loading (PyTorch / C extensions in virtual environments)
if sys.platform == "win32":
    try:
        site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
        torch_lib = site_packages / "torch" / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()  # Fallback to system environment

# Directory paths
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
UPLOADS_DIR = BASE_DIR / "uploads"
MEMORY_FILE = DATA_DIR / "memory.json"

# Ensure necessary directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

# Embedding & Vector Search Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

def is_gemini_configured() -> bool:
    """Checks if a valid Gemini API key is configured in the server environment."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    return bool(api_key and api_key != "your_gemini_api_key_here")

# Optional API client helper
def get_gemini_client():
    """
    Initializes and returns the Google Generative AI client or model.
    Raises ValueError if the API key is not configured.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please ensure the GEMINI_API_KEY environment variable "
            "is configured on the server."
        )

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai

def generate_gemini_text(prompt: str, system_instruction: str = None) -> str:
    """
    Executes a prompt against the configured Gemini model with comprehensive error handling.
    If the requested model is unavailable, it automatically tries fallback models
    so the application never crashes.
    """
    try:
        genai = get_gemini_client()
        model_name = os.getenv("GEMINI_MODEL", GEMINI_MODEL_NAME).strip()

        kwargs = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        # Ordered list of models to try (primary first, then fallbacks)
        model_candidates = [
            model_name,
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
        ]
        # Remove duplicates while preserving order
        seen = set()
        model_candidates = [m for m in model_candidates if not (m in seen or seen.add(m))]

        # Merge system_instruction into the prompt to avoid kwarg compatibility issues
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n---\n\n{prompt}"

        response = None
        last_error = None
        for try_model in model_candidates:
            try:
                model = genai.GenerativeModel(model_name=try_model)
                response = model.generate_content(
                    full_prompt,
                    request_options={"timeout": 60},
                )
                break
            except Exception as api_err:
                last_error = api_err
                err_str = str(api_err)
                if "404" in err_str or "NotFound" in err_str or "not found" in err_str.lower():
                    continue
                else:
                    raise api_err

        if not response:
            if last_error:
                raise last_error
            return "No response received from Gemini model. Please try again."

        # Safely access response text — may raise if blocked by safety filters
        try:
            text = response.text
        except ValueError:
            # Response was blocked by safety filters
            return "The response was blocked by content safety filters. Please rephrase your question."

        if not text:
            return "No response received from Gemini model. Please try again."

        return text.strip()
    except ValueError as ve:
        # Missing API Key
        raise ve
    except Exception as e:
        err_msg = str(e)
        model_name = os.getenv("GEMINI_MODEL", GEMINI_MODEL_NAME).strip()
        if "404" in err_msg or "NotFound" in err_msg or "models/" in err_msg:
            raise RuntimeError(
                f"The configured Gemini model '{model_name}' was not found or is unavailable for your API key.\n"
                f"Error details: {err_msg}\n\n"
                f"Tip: Update GEMINI_MODEL in your .env file (e.g. GEMINI_MODEL=gemini-3.8-flash, gemini-3.5-flash, or gemini-flash-latest)."
            ) from e
        elif "API_KEY_INVALID" in err_msg or ("400" in err_msg and "api key" in err_msg.lower()):
            raise RuntimeError(
                "Invalid Gemini API Key provided. Please verify your GEMINI_API_KEY in the .env file."
            ) from e
        elif "ResourceExhausted" in err_msg or "429" in err_msg:
            raise RuntimeError(
                "Gemini API rate limit exceeded or quota exhausted. Please wait a moment before trying again."
            ) from e
        else:
            raise RuntimeError(
                f"Gemini API error ({type(e).__name__}): {err_msg}\n"
                f"Model: '{model_name}'. Check your internet connection or model availability in .env."
            ) from e
