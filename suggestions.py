# suggestions.py
from __future__ import annotations
import os, json
from typing import Dict, Iterable, List, Optional

# Classifier label set (no remapping)
CLASSIFIER_LABELS = {"joy", "boredom", "hunger", "aggressivity", "sadness"}

# Safety-first, concise guidance per classifier label
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

# ----------------- public API -----------------

def llm_suggestion(
    *,
    top_label: str,
    scores: Optional[Iterable[Dict[str, float]]] = None,
    context: Optional[str] = None,
    # kept for API compatibility; when True we still treat labels as classifier labels
    classifier_space: bool = True,
) -> Dict[str, str]:
    """
    Build concise user-facing guidance in *classifier label space*.
    Returns: {"state": str, "suggestion": str, "reason": str}
    """
    top = str(top_label).strip().lower()
    if top not in CLASSIFIER_LABELS:
        top = "unknown"

    probs = {}
    if scores:
        probs = _normalize_probs(_scores_to_dict(scores))
        # Drop any non-classifier keys
        probs = {k: v for k, v in probs.items() if k in CLASSIFIER_LABELS}
        probs = _normalize_probs(probs)
    confidence = float(probs.get(top, 0.0)) if probs else 0.0

    # Optional Anthropic path; graceful fallback to RULES on any issue
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)

            label_doc = (
                "- joy: positive/playful arousal\n"
                "- boredom: under-stimulation\n"
                "- hunger: food expectation/need\n"
                "- aggressivity: reactivity/guarding risk\n"
                "- sadness: low arousal/whine\n"
            )
            system = (
                "You assist a dog-bark classifier. Respond with compact JSON keys "
                "{state, suggestion, reason}. One sentence each. Safety-first tone."
            )
            user = (
                f"Context: {context or 'N/A'}\n"
                f"Top label (classifier space): {top}\n"
                f"Label probabilities (classifier space): "
                f"{json.dumps(probs or {}, separators=(',',':'))}\n\n"
                f"Label doc:\n{label_doc}\n\n"
                "Return ONLY JSON with keys exactly: state, suggestion, reason."
            )
            msg = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=200,
                temperature=0.2,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(getattr(p, "text", "") for p in msg.content)
            try:
                data = json.loads(text)
                if isinstance(data, dict) and all(k in data for k in ("state", "suggestion", "reason")):
                    return data
            except Exception:
                pass
        except Exception:
            pass

    rule = RULES.get(top, RULES["unknown"])
    conf_pct = f"{int(round(confidence * 100))}%" if confidence else "low"
    reason = f"Top label '{top}' with {conf_pct} confidence based on acoustic features."
    return {"state": rule["state"], "suggestion": rule["suggestion"], "reason": reason}

# Helpers you can import in main.py
def normalize_classifier_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """Public helper to sanitize/normalize classifier-space probabilities."""
    p = _normalize_probs(probs or {})
    p = {k: v for k, v in p.items() if k in CLASSIFIER_LABELS}
    return _normalize_probs(p)

__all__ = ["llm_suggestion", "normalize_classifier_probs", "CLASSIFIER_LABELS"]
