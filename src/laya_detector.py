"""Laya Decision Engine Adapter for Deep Phishing Analysis.

Integrates Convai Innovations' Laya (ModernBERT-large System 1 Decision Model),
providing calibrated probability decisions for URLs, emails, and HTML context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LAYA_AGENT = None
_LAYA_LOAD_FAILED = False


def get_laya_agent() -> Any:
    """Lazy-load the Laya decision agent."""
    global _LAYA_AGENT, _LAYA_LOAD_FAILED
    if _LAYA_AGENT is not None:
        return _LAYA_AGENT
    if _LAYA_LOAD_FAILED:
        return None

    import os
    # Do not auto-download large weights unless explicitly enabled
    if not os.getenv("LAYA_ENABLE_AUTODOWNLOAD"):
        return None

    try:
        import laya
        logger.info("Loading Laya decision engine (convaiinnovations/laya)...")
        _LAYA_AGENT = laya.load("convaiinnovations/laya")
        return _LAYA_AGENT
    except Exception as exc:
        logger.warning("Could not load Laya model (%s). Fallback mode active.", exc)
        _LAYA_LOAD_FAILED = True
        return None


def is_laya_available() -> bool:
    """Check if the Laya model is currently loaded or loadable."""
    return get_laya_agent() is not None


def analyze_url_with_laya(url: str) -> Dict[str, Any]:
    """Perform deep semantic classification of a URL using Laya."""
    agent = get_laya_agent()
    if agent is None:
        return {
            "available": False,
            "error": "Laya model weights not loaded or environment lacks sufficient memory for 421M parameter model.",
            "verdict": None,
            "phishing_probability": None,
        }

    try:
        state = f"Target URL: {url}"
        questions = {
            "is_phishing": {
                "type": "noul",
                "instructions": (
                    "Is this URL a deceptive phishing, credential harvesting, or malware link, "
                    "or is it an authentic, legitimate web URL?"
                ),
                "criteria": {
                    "true": "phishing, credential theft, fake brand impersonation, or scam",
                    "false": "legitimate web link, authentic corporate, open-source, or educational resource",
                },
            },
            "brand_impersonation": {
                "type": "noul",
                "instructions": "Does this URL mimic or impersonate a well-known brand, bank, or service using typosquatting or fake subdomains?",
            },
        }

        result = agent.predict(state, questions)
        is_phishing_prob = float(result.get("is_phishing", {}).get("prob", 0.5))
        is_phish_bool = bool(result.get("is_phishing", {}).get("value", is_phishing_prob >= 0.5))
        brand_impersonation = bool(result.get("brand_impersonation", {}).get("value", False))

        return {
            "available": True,
            "model": "convaiinnovations/laya (ModernBERT-large)",
            "verdict": "PHISHING" if is_phish_bool else "LEGITIMATE",
            "phishing_probability": round(is_phishing_prob, 4),
            "confidence": round(max(is_phishing_prob, 1.0 - is_phishing_prob), 4),
            "brand_impersonation_detected": brand_impersonation,
            "raw_result": result,
        }
    except Exception as exc:
        logger.error("Laya analysis failed: %s", exc)
        return {
            "available": False,
            "error": str(exc),
            "verdict": None,
            "phishing_probability": None,
        }


def analyze_email_with_laya(email_text: str) -> Dict[str, Any]:
    """Perform deep semantic classification of email content using Laya presets."""
    agent = get_laya_agent()
    if agent is None:
        return {
            "available": False,
            "error": "Laya model not initialized.",
        }

    try:
        import laya.presets
        questions = laya.presets.email_questions()
        state = {"body": email_text}
        result = agent.predict(state, questions)
        return {
            "available": True,
            "model": "convaiinnovations/laya",
            "result": result,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
