#!/usr/bin/env python3
"""
Utility helpers for Google Gemini (generativeai) usage.

- Automatically loads API key from .env (GEMINI_API_KEY or GOOGLE_API_KEY).
- Provides simple text generation helper with consistent (ok, result) tuple.
- Supports file uploads for document analysis.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _manual_load_env(path: str) -> None:
    """Very small .env loader if python-dotenv is missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _load_env() -> None:
    """Load .env if python-dotenv is available; otherwise minimal manual loader."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        # Fallback: manual load from project root .env
        root_env = os.path.join(os.path.dirname(__file__), ".env")
        _manual_load_env(root_env)


def get_gemini_api_key() -> Optional[str]:
    """Return the Gemini API key from environment or .env."""
    _load_env()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key if key and key.strip() else None


def _ensure_client() -> Any:
    """
    Configure Gemini client.
    Uses new SDK (google.genai) which is the current recommended approach.
    Returns a tuple (client, flavor) where flavor is "new" or "old".
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not found. "
            "Set it in your environment or .env file."
        )

    # Use new SDK (google.genai)
    try:
        import google.genai as genai_new  # type: ignore
        client = genai_new.Client(api_key=api_key)
        return (client, "new")
    except Exception as e:
        raise RuntimeError(
            "google-genai is required.\n"
            "Install with:\n"
            "  pip install --upgrade google-genai"
        ) from e


def generate_text(
    prompt: str,
    model: str = "gemini-2.0-flash",
    generation_config: Optional[Dict[str, Any]] = None,
    safety_settings: Optional[Any] = None,
) -> Tuple[bool, Any]:
    """
    Generate text with Gemini.

    Model names:
    - For new SDK (google.genai): "models/gemini-2.0-flash", "models/gemini-2.5-flash", etc.
    - For old SDK (google.generativeai): "gemini-1.5-flash" or "gemini-1.5-pro"

    Returns:
        (ok, text_or_error)
    """
    try:
        client, flavor = _ensure_client()
        if flavor == "new":
            # google.genai client - ensure model has "models/" prefix
            model_to_try = model
            if not model.startswith("models/"):
                model_to_try = f"models/{model}"
            
            # Map common aliases to actual model names
            alias_map = {
                "gemini-1.5-flash": "models/gemini-2.0-flash",
                "gemini-1.5-pro": "models/gemini-2.5-pro",
                "gemini-2.0-flash": "models/gemini-2.0-flash",
                "gemini-2.5-flash": "models/gemini-2.5-flash",
                "gemini-2.5-pro": "models/gemini-2.5-pro",
            }
            if model in alias_map:
                model_to_try = alias_map[model]
            elif model_to_try not in alias_map.values():
                # If it's not a known model, keep the models/ prefix version
                pass
            
            try:
                try:
                    resp = client.models.generate_content(
                        model=model_to_try,
                        contents=prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                    )
                except TypeError:
                    # Fallback: some versions don't accept generation_config/safety_settings
                    resp = client.models.generate_content(
                        model=model_to_try,
                        contents=prompt,
                    )
            except Exception as e:
                # If model not found, try listing available models
                if "404" in str(e) or "not found" in str(e).lower():
                    try:
                        models = client.models.list()
                        available = [m.name for m in models if hasattr(m, "name")]
                        return False, f"Model {model_to_try} not found. Available models: {available[:10]}"
                    except Exception:
                        pass
                raise
            # New SDK response shape: resp.candidates[0].content.parts or resp.text if available
            if hasattr(resp, "text") and isinstance(resp.text, str):
                return True, resp.text
            try:
                return True, resp.candidates[0].content.parts[0].text  # type: ignore
            except Exception:
                return True, str(resp)
        else:
            # Deprecated google.generativeai
            mdl = client.GenerativeModel(model)
            resp = mdl.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            if hasattr(resp, "text") and isinstance(resp.text, str):
                return True, resp.text
            return True, str(resp)
    except Exception as e:
        return False, str(e)


def generate_text_with_url(
    prompt: str,
    file_url: str,
    mime_type: str = "application/pdf",
    model: str = "gemini-2.0-flash",
    generation_config: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Tuple[bool, Any]:
    """
    Generate text with Gemini using a file URL (Gemini fetches the file itself).
    Includes retry logic for 503 errors (overloaded) with exponential backoff.
    
    Args:
        prompt: Text prompt/instructions
        file_url: URL to the file (PDF, image, etc.)
        mime_type: MIME type of the file (default: application/pdf)
        model: Model name
        generation_config: Optional generation config
        max_retries: Maximum number of retries for 503 errors
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        (ok, text_or_error)
    """
    client, flavor = _ensure_client()
    
    # Use new SDK (google.genai) with Part.from_uri() for HTTP URLs
    from google.genai.types import Part  # type: ignore
    
    model_to_try = model
    if not model.startswith("models/"):
        model_to_try = f"models/{model}"
    
    alias_map = {
        "gemini-1.5-flash": "models/gemini-2.0-flash",
        "gemini-1.5-pro": "models/gemini-2.5-pro",
        "gemini-2.0-flash": "models/gemini-2.0-flash",
        "gemini-2.5-flash": "models/gemini-2.5-flash",
        "gemini-2.5-pro": "models/gemini-2.5-pro",
    }
    if model in alias_map:
        model_to_try = alias_map[model]
    
    # Create file part from HTTP URL using Part.from_uri()
    file_part = Part.from_uri(
        file_uri=file_url,
        mime_type=mime_type
    )
    
    # Retry logic for 503 errors
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_to_try,
                contents=[file_part, prompt],
            )
            
            # Extract text from response
            if hasattr(resp, "text") and isinstance(resp.text, str):
                return True, resp.text
            try:
                return True, resp.candidates[0].content.parts[0].text  # type: ignore
            except Exception:
                return True, str(resp)
                
        except Exception as e:
            error_str = str(e)
            last_error = e
            
            # Check if it's a 503 error (overloaded) - retry with backoff
            if "503" in error_str or "overloaded" in error_str.lower() or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    delay = min(delay, 60.0)  # Cap at 60 seconds
                    time.sleep(delay)
                    continue
                else:
                    return False, f"503 UNAVAILABLE after {max_retries} retries: {error_str}"
            else:
                # For other errors, don't retry
                return False, str(e)
    
    return False, f"Failed after {max_retries} attempts: {last_error}"


def upload_file_to_gemini(file_path: Path, mime_type: str = "application/pdf") -> Tuple[bool, Any]:
    """
    Upload file to Gemini Files API and return uploaded file object.
    This is fast - uploads file separately, then we reference it by ID.
    Handles Unicode filenames properly by reading as bytes.
    
    Returns:
        (ok, uploaded_file_object_or_error)
    """
    try:
        client, flavor = _ensure_client()
        
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        
        # Read file as bytes to avoid Unicode path encoding issues
        import io
        file_data = file_path.read_bytes()
        file_io = io.BytesIO(file_data)
        
        # Use new SDK file upload with UploadFileConfig
        from google.genai import types  # type: ignore
        
        uploaded_file = client.files.upload(
            file=file_io,
            config=types.UploadFileConfig(mime_type=mime_type)
        )
        
        # Wait for file to be processed (usually instant)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(0.5)
            uploaded_file = client.files.get(name=uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            return False, f"File upload failed: {uploaded_file.state.name}"
        
        return True, uploaded_file  # Return file object
    except Exception as e:
        return False, str(e)


def generate_text_with_file(
    prompt: str,
    file_path: Path,
    mime_type: str = "application/pdf",
    model: str = "gemini-2.0-flash",
    generation_config: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Tuple[bool, Any]:
    """
    Generate text with Gemini using a file attachment.
    Uses fast file upload API - uploads file first, then references by ID.
    Includes retry logic for 503 errors (overloaded) with exponential backoff.
    
    Args:
        prompt: Text prompt/instructions
        file_path: Path to file to upload (PDF, image, etc.)
        mime_type: MIME type of the file (default: application/pdf)
        model: Model name
        generation_config: Optional generation config
        max_retries: Maximum number of retries for 503 errors
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        (ok, text_or_error)
    """
    client, flavor = _ensure_client()
    
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    # Upload file first (fast - uses Gemini's file upload API)
    ok_upload, uploaded_file = upload_file_to_gemini(file_path, mime_type)
    if not ok_upload:
        return False, f"File upload failed: {uploaded_file}"
    
    model_to_try = model
    if not model.startswith("models/"):
        model_to_try = f"models/{model}"
    
    alias_map = {
        "gemini-1.5-flash": "models/gemini-2.0-flash",
        "gemini-1.5-pro": "models/gemini-2.5-pro",
        "gemini-2.0-flash": "models/gemini-2.0-flash",
        "gemini-2.5-flash": "models/gemini-2.5-flash",
        "gemini-2.5-pro": "models/gemini-2.5-pro",
    }
    if model in alias_map:
        model_to_try = alias_map[model]
    
    # Retry logic for 503 errors
    last_error = None
    for attempt in range(max_retries):
        try:
            # Pass uploaded file directly (as shown in example) - much faster!
            resp = client.models.generate_content(
                model=model_to_try,
                contents=[uploaded_file, prompt],  # File first, then prompt
            )
            
            # Extract text from response
            if hasattr(resp, "text") and isinstance(resp.text, str):
                return True, resp.text
            try:
                return True, resp.candidates[0].content.parts[0].text  # type: ignore
            except Exception:
                return True, str(resp)
                
        except Exception as e:
            error_str = str(e)
            last_error = e
            
            # Check if it's a 503 error (overloaded) - retry with backoff
            if "503" in error_str or "overloaded" in error_str.lower() or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    delay = min(delay, 60.0)  # Cap at 60 seconds
                    time.sleep(delay)
                    continue
                else:
                    return False, f"503 UNAVAILABLE after {max_retries} retries: {error_str}"
            else:
                # For other errors, don't retry
                return False, str(e)
    
    return False, f"Failed after {max_retries} attempts: {last_error}"





