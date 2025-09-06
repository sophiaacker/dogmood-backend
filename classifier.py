# ALL MOCK RIGHT NOW
import random
from typing import Dict
from suggestions import CLASSES

def classify_bark(audio_path: str) -> Dict:
    # Deterministic "random" based on file path bytes
    seed = sum(bytearray(audio_path.encode()))
    rnd = random.Random(seed)
    probs = {c: 0.0 for c in CLASSES}
    main = rnd.choice(CLASSES[:-1])  # avoid 'unknown' most of the time
    main_conf = round(rnd.uniform(0.6, 0.95), 2)
    probs[main] = main_conf
    remain = max(0.0, 1 - main_conf)
    others = [c for c in CLASSES if c != main]
    for c in others:
        probs[c] = round(remain / len(others), 2)
    label = max(probs, key=probs.get)
    return {"label": label, "confidence": probs[label], "probs": probs}
