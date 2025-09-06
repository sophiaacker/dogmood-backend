# suggestions.py
from __future__ import annotations
import os, json, re
from typing import Dict, Iterable, List, Optional

# ----------------- debug helpers -----------------
DEBUG = os.getenv("SUGGESTIONS_DEBUG") == "1"
def _dbg(*a):
    if DEBUG:
        print("[SUGGESTIONS]", *a)

# Try to load .env locally so you don't forget to export vars during dev
try:
    from dotenv import load_dotenv  # optional; if not installed it's fine
    load_dotenv()
except Exception:
    pass

# ----------------- label space -------------------
# We stay in *classifier* label space end-to-end.
CLASSIFIER_LABELS = {"joy", "boredom", "hunger", "aggressivity", "sadness"}

# Safety-first, concise rule fallback per classifier label
RULES: Dict[str, Dict[str, str]] = {
    "joy": {
        "state": "The vocalization suggests a joyful, playful mood.",
        "suggestion": "Offer a quick play session or toy to positively channel the energy.",
    },
    "boredom": {
        "state": "The vocalization suggests boredom or under-stimulation.",
        "suggestion": "Provide enrichment: puzzle feeder, snuffle mat, or a short training game.",
    },
    "hunger": {
        "state": "The vocalization may indicate hunger or food expectation.",
        "suggestion": "Check feeding schedule and portions; avoid reinforcing demand barking; consult a vet if persistent.",
    },
    "aggressivity": {
        "state": "The vocalization suggests increased reactivity or aggressivity.",
        "suggestion": "Do not approach; reduce triggers and give space. Seek a qualified trainer if it persists.",
    },
    "sadness": {
        "state": "The vocalization may reflect sadness or low arousal.",
        "suggestion": "Offer calm reassurance and gentle engagement; monitor context and consult a vet if ongoing.",
    },
    "unknown": {
        "state": "The signal is unclear.",
        "suggestion": "Try another recording in a quieter environment, slightly closer but non-intrusive.",
    },
}

# ----------------- utils -----------------

def _normalize_probs(raw: Dict[str, float]) -> Dict[str, float]:
    """Return finite probs that sum to 1 (best effort)."""
    if not raw:
        return {}
    clean: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                continue
            clean[str(k).strip().lower()] = f
        except Exception:
            continue
    if not clean:
        return {}
    total = sum(clean.values())
    if total <= 0:
        n = len(clean)
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}

def _scores_to_dict(scores: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Convert [{'label': str, 'score': float}, ...] → {label: score}."""
    d: Dict[str, float] = {}
    for s in scores or []:
        lbl = str(s.get("label", "unknown")).strip().lower()
        val = float(s.get("score", 0.0))
        d[lbl] = d.get(lbl, 0.0) + val
    return d

def _sorted_scores_from_probs(probs: Dict[str, float]) -> List[Dict[str, float]]:
    return sorted(
        ({"label": k, "score": float(v)} for k, v in probs.items()),
        key=lambda x: x["score"],
        reverse=True,
    )

# ----------------- anthropic client -----------------

def _try_llm(top: str, probs: Dict[str, float], context: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Attempt an Anthropic call. Return dict({state,suggestion,reason}) or None on any issue.
    Includes robust JSON extraction and debug logs.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        _dbg("ANTHROPIC_API_KEY not set; using RULES fallback.")
        return None

    try:
        from anthropic import Anthropic
    except Exception as e:
        _dbg("anthropic import failed:", repr(e))
        return None

    # Configurable model, temperature, timeout via env; sensible defaults
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    try:
        temperature = float(os.getenv("SUGGESTIONS_TEMPERATURE", "0.2"))
    except Exception:
        temperature = 0.2
    try:
        timeout = float(os.getenv("SUGGESTIONS_TIMEOUT_SEC", "20"))
    except Exception:
        timeout = 20.0

    client = Anthropic(api_key=api_key)
    # Note: older anthropic SDKs may not support .with_options; we'll rely on default timeout behavior.
    _dbg("LLM model:", model, "temp:", temperature, "timeout:", timeout)

    label_doc = (
        "- joy: positive/playful arousal\n"
        "- boredom: under-stimulation\n"
        "- hunger: food expectation/need\n"
        "- aggressivity: reactivity/guarding risk\n"
        "- sadness: low arousal/whine\n"
    )

    # FIXED: the original prompt had adjacent strings without spaces → merged words → worse parsing.
    system = (
        "You are dog behavior analyst and veterinarian "
        "Given classifier results from a bark, analyze the dog's moods. "
        "Focus on the moods with the highest levels of expression. Based on the breakdown of moods the dog is experiencing, recommend actionable items that a dog owner can do to help with the current mood. "
        "Avoid recommendations that only humans can do (e.g. taking deep breaths, journaling reading)"
        "Target approaches that focus on humans helping out. For example, if a dog is anxious, you may expect taking them on a walk."
        "Include product recommendations when relevant. You may recommend Kong peanut butter toys for example or certain teeth cleaning chews."
        "Respond ONLY as JSON with keys {state, suggestion, reason}. "
    )

    user = (
        f"Context: {context or 'N/A'}\n"
        f"Top label (classifier space): {top}\n"
        f"Label probabilities (classifier space): {json.dumps(probs or {}, separators=(',',':'))}\n\n"
        f"Label guide:\n{label_doc}\n\n"
        "Return ONLY JSON with keys exactly: state, suggestion, reason."
    )

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        _dbg("Anthropic request error:", repr(e))
        return None

    # ---- robust JSON extraction ----
    text = "".join(getattr(p, "text", "") for p in getattr(msg, "content", []) if hasattr(p, "text"))
    _dbg("RAW:", repr(text[:400]) + ("..." if len(text) > 400 else ""))

    data = None
    # 1) direct parse
    try:
        data = json.loads(text)
    except Exception:
        pass

    # 2) fenced blocks ```json ... ```
    if data is None:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception as e:
                _dbg("json.loads fenced failed:", repr(e))

    # 3) first {...} blob
    if data is None:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception as e:
                _dbg("json.loads brace-scan failed:", repr(e))

    if isinstance(data, dict) and all(k in data for k in ("state", "suggestion", "reason")):
        _dbg("LLM JSON accepted.")
        return data

    _dbg("LLM returned non-JSON or wrong keys; falling back to RULES.")
    return None

# ----------------- public API -----------------

def llm_suggestion(
    *,
    top_label: str,
    scores: Optional[Iterable[Dict[str, float]]] = None,
    context: Optional[str] = None,
    # kept for API compatibility; labels are classifier labels either way
    classifier_space: bool = True,
) -> Dict[str, str]:
    """
    Build concise user-facing guidance in *classifier label space*.
    Returns: {"state": str, "suggestion": str, "reason": str}
    """
    top = str(top_label).strip().lower()
    probs: Dict[str, float] = {}

    if scores:
        probs = _normalize_probs(_scores_to_dict(scores))
        # Keep only known classifier labels
        probs = {k: v for k, v in probs.items() if k in CLASSIFIER_LABELS}
        probs = _normalize_probs(probs)

    if top not in CLASSIFIER_LABELS:
        # If we got an unknown top but we have probabilities, pick the argmax;
        # otherwise, mark as unknown.
        top = max(probs, key=probs.get) if probs else "unknown"

    confidence = float(probs.get(top, 0.0)) if probs else 0.0

    # Try LLM; if it fails for any reason, we fall back to RULES
    llm_out = _try_llm(top=top, probs=probs, context=context)
    if llm_out is not None:
        return llm_out

    # Fallback
    rule = RULES.get(top, RULES["unknown"])
    conf_pct = f"{int(round(confidence * 100))}%" if confidence else "low"
    reason = f"Top label '{top}' with {conf_pct} confidence based on acoustic features."
    return {"state": rule["state"], "suggestion": rule["suggestion"], "reason": reason}

# Helper you can import in main.py
def normalize_classifier_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """Public helper to sanitize/normalize classifier-space probabilities."""
    p = _normalize_probs(probs or {})
    p = {k: v for k, v in p.items() if k in CLASSIFIER_LABELS}
    return _normalize_probs(p)

__all__ = ["llm_suggestion", "normalize_classifier_probs", "CLASSIFIER_LABELS"]
