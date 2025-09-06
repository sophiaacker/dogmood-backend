# suggestions.py
from __future__ import annotations
import os, json
from typing import Dict, List, Optional, TypedDict

# ===== Static fallbacks (used ONLY if the API fails) =====
SUGGESTIONS: Dict[str, str] = {
    "playful":  "Play tug or fetch for 5–10 minutes. Offer a toy.",
    "anxious":  "Create distance from the trigger and use a calm cue. Exercise your dog",
    "aggressive": "Do NOT approach. Remove stimuli; consider a trainer.",
    "bored":    "Try a snuffle mat or 10-minute puzzle feeder.",
    "unknown":  "Re-record in a quieter room for 3–5 seconds."
}
CLASSES: List[str] = ["playful", "anxious", "aggressive", "bored", "unknown"]

class ClassScore(TypedDict):
    label: str
    score: float  # 0..1

class SuggestionJSON(TypedDict):
    state: str
    suggestion: str
    reason: str

DEBUG = os.getenv("DOGSUGGEST_DEBUG", "0") == "1"

# You can force a model via env: DOGSUGGEST_MODEL="claude-3-5-haiku-latest"
ENV_MODEL = os.getenv("DOGSUGGEST_MODEL")
MODEL_CANDIDATES: List[str] = [
    m for m in [
        ENV_MODEL,
        "claude-3-5-haiku-latest",
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
    ] if m
]

# ⛔️ Ban phrases that echo your baselines
BANNED_PHRASES = [
    "Create distance from the trigger",
    "Offer a toy",
    "Try a snuffle mat",
    "puzzle feeder",
    "Re-record in a quieter room",
]

SYSTEM_PROMPT = (
    "You are a canine veterinarian and behavior micro-coach.\n"
    "Goal: combine multiple emotional labels into ONE plan.\n"
    "Requirements:\n"
    "Write a plain-English STATE describing the overall mood, explicitly naming the top 1–2 emotions (e.g., 'mixed playfulness and anxiety').\n"
    "Write ONE SUGGESTION that is a composite action plan that addresses each prominent emotion, with 2–3 short clauses separated by commas or semicolons.\n"
    "  – Include one immediate regulation step (distance/downshift/soothing), AND one engagement/enrichment step (play, sniff, chew, scatter feed).\n"
    "  – Tailor to provided context (location, trigger, time limits).\n"
    "  - Suggestions should draw from safe, realistic dog enrichment strategies such as walks, sniffing games, chew toys, puzzle feeders, hide-and-seek, or short training exercises, and must avoid impossible or human-only activities (e.g. breathing, reading)."
    "Write a short REASON explaining how the plan helps both emotions.\n"
    "Hard rules:\n"
    "• Do NOT copy or paraphrase any default baseline text; avoid these phrases entirely: "
    + "; ".join(f"'{p}'" for p in BANNED_PHRASES) + ".\n"
    "• Avoid generic fluff. Be specific and safe. For 'aggressive', prioritize safety then professional help if it persists.\n"
    'Output STRICT JSON only as: {"state":"string","suggestion":"string","reason":"string"}'
)

# ===== Anthropic client =====
# pip install anthropic
# export ANTHROPIC_API_KEY=sk-ant-...
from anthropic import Anthropic
from anthropic import NotFoundError, PermissionDeniedError, RateLimitError
_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def _build_user_prompt(
    top_label: str,
    scores: Optional[List[ClassScore]] = None,
    context: Optional[str] = None,
) -> str:
    """
    Intentionally omits baseline texts to avoid anchoring/copying.
    Provides ranked labels + context and asks for a composite plan.
    """
    payload = {
        "top_label": top_label,
        "scores": scores,          # e.g. [{"label":"anxious","score":0.41}, ...]
        "user_context": context,   # free text, optional
        "style_guidance": {
            "structure": "state + single composite suggestion + reason",
            "lengths": {"suggestion_words_max": 25, "reason_words_max": 20},
            "include_emotions": "explicitly name top 1–2 emotions in 'state'",
            "cover_each_emotion": True
        }
    }
    return "Classifier result and context:\n" + json.dumps(payload, ensure_ascii=False)

def _claude_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
) -> dict:
    resp = _client.messages.create(
        model=model,
        max_tokens=220,
        temperature=temperature,  # encourage creativity
        top_p=top_p,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=timeout_seconds,
    )
    text = resp.content[0].text if resp.content else ""
    if DEBUG:
        print(f"[dogsuggest] model={model} raw_text:", repr(text))
    return json.loads(text)

def _fallback_json(top_label: str) -> SuggestionJSON:
    base = SUGGESTIONS.get(top_label, SUGGESTIONS["unknown"])
    label_nice = top_label if top_label in SUGGESTIONS else "unknown"
    return {
        "state": f"Your dog likely shows primarily {label_nice} behavior.",
        "suggestion": base,
        "reason": "Using baseline guidance due to model error."
    }

def llm_suggestion(
    top_label: str,
    scores: Optional[List[ClassScore]] = None,
    context: Optional[str] = None,
    *,
    temperature: float = 0.6,       # ↑ a bit for more variety
    top_p: float = 0.9,             # sample more creatively but safely
    timeout_seconds: float = 8.0,
    system_prompt: Optional[str] = None,
) -> SuggestionJSON:
    """
    Returns {"state","suggestion","reason"} from Anthropic Claude.
    Uses creative sampling and forbids baseline regurgitation.
    """
    if top_label not in SUGGESTIONS:
        top_label = "unknown"

    user_prompt = _build_user_prompt(top_label, scores, context)
    sys_prompt = system_prompt or SYSTEM_PROMPT

    last_error: Optional[Exception] = None
    for model in MODEL_CANDIDATES:
        try:
            data = _claude_json(model, sys_prompt, user_prompt, temperature, top_p, timeout_seconds)

            # --- strict parse & normalization ---
            state = (data.get("state") or "").strip()
            suggestion = (data.get("suggestion") or "").strip()
            reason = (data.get("reason") or "").strip()

            if not state or not suggestion:
                raise ValueError("Missing 'state' or 'suggestion' in JSON")

            # Soft guard: discourage banned baseline phrasing
            lower = suggestion.lower()
            if any(p.lower() in lower for p in BANNED_PHRASES):
                raise ValueError("Suggestion resembled a banned baseline phrase")

            return {"state": state, "suggestion": suggestion, "reason": reason}

        except NotFoundError as e:
            last_error = e
            if DEBUG:
                print(f"[dogsuggest] Model not found, trying next: {model} -> {repr(e)}")
            continue
        except PermissionDeniedError as e:
            last_error = e
            if DEBUG:
                print(f"[dogsuggest] Permission denied for model {model}: {repr(e)}")
            continue
        except RateLimitError as e:
            last_error = e
            if DEBUG:
                print(f"[dogsuggest] Rate limited on model {model}: {repr(e)}")
            break
        except Exception as e:
            last_error = e
            if DEBUG:
                import traceback
                print(f"[dogsuggest] Error with model {model}: {repr(e)}")
                traceback.print_exc()
            # Try next candidate only for model-not-found/permission; otherwise break
            break

    if DEBUG and last_error:
        print("[dogsuggest] Falling back to baseline JSON due to:", repr(last_error))
    return _fallback_json(top_label)

def suggestion_for_label(label: str) -> SuggestionJSON:
    return llm_suggestion(top_label=label)

# ===== CLI quick test =====
if __name__ == "__main__":
    ranked = [
        {"label": "anxious", "score": 0.41},
        {"label": "bored", "score": 0.37},
        {"label": "unknown", "score": 0.12},
    ]
    result = llm_suggestion(
        top_label="anxious",
        scores=ranked,
        context="Whining at the window after a loud truck passed; owner has 5 minutes.",
        temperature=0.65,
        top_p=0.9,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
