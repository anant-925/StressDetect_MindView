"""
utils/llm_reward.py
===================
LLM-as-judge reward model for the RL feedback loop.

The module queries an external LLM (OpenAI GPT or Google Gemini) to evaluate
whether the stress-detection model's prediction for a piece of text is
reasonable.  The LLM acts as an automated teacher: it returns +1 when the
prediction seems correct and -1 when it seems wrong.

Configuration
-------------
Set **one** of the following environment variables:

- ``OPENAI_API_KEY``   — enables the OpenAI provider (default model: gpt-4o-mini)
- ``GEMINI_API_KEY``   — enables the Google Gemini provider

If neither key is set the function returns ``None`` gracefully so that the
system continues to work with user feedback alone.

Provider selection
------------------
Pass ``provider="openai"`` or ``provider="gemini"`` explicitly, or leave it as
``"auto"`` (default) to use whichever API key is detected in the environment
(OpenAI takes priority when both keys are set).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Prompt template — keep it short so cheap/fast models respond reliably.
_PROMPT_TEMPLATE = """You are evaluating an AI stress-detection system.

Text submitted by the user:
"{text}"

The system predicted a stress probability of {prediction:.2f} (threshold 0.5,
so the label is "{label}").

Is this prediction reasonable given the text?
Answer with exactly one word: YES or NO."""


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call the OpenAI Chat Completions API and return the raw text reply."""
    import openai  # local import — optional dependency

    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def _call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """Call the Google Generative AI (Gemini) API and return the raw text reply."""
    import google.generativeai as genai  # local import — optional dependency

    api_key = os.environ.get("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 5, "temperature": 0.0},
    )
    return response.text or ""


def get_llm_reward(
    text: str,
    prediction: float,
    provider: str = "auto",
) -> Optional[int]:
    """Ask an LLM whether the stress prediction is reasonable.

    Parameters
    ----------
    text : str
        The user's input text that was analysed.
    prediction : float
        Raw stress probability from the model (0–1).
    provider : str
        ``"auto"`` (default), ``"openai"``, or ``"gemini"``.

    Returns
    -------
    int | None
        ``+1`` if the LLM says the prediction is reasonable,
        ``-1`` if the LLM says it is not,
        ``None`` if no API key is configured or the call fails.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    # Resolve provider
    if provider == "auto":
        if openai_key:
            provider = "openai"
        elif gemini_key:
            provider = "gemini"
        else:
            logger.debug(
                "No LLM API key found (OPENAI_API_KEY / GEMINI_API_KEY); "
                "skipping LLM reward."
            )
            return None

    if provider == "openai" and not openai_key:
        logger.debug("OPENAI_API_KEY not set; skipping LLM reward.")
        return None
    if provider == "gemini" and not gemini_key:
        logger.debug("GEMINI_API_KEY not set; skipping LLM reward.")
        return None

    label = "stressed" if prediction >= 0.5 else "not stressed"
    prompt = _PROMPT_TEMPLATE.format(
        text=text[:500],  # Truncate to avoid large token usage
        prediction=prediction,
        label=label,
    )

    try:
        if provider == "openai":
            raw = _call_openai(prompt)
        else:
            raw = _call_gemini(prompt)

        answer = raw.strip().upper()
        if "YES" in answer:
            return 1
        if "NO" in answer:
            return -1

        logger.warning("Unexpected LLM response %r; treating as None.", raw)
        return None

    except Exception as exc:
        logger.warning("LLM reward call failed (%s); continuing without it.", exc)
        return None
