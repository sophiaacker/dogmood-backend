
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, argparse, wave, struct, numpy as np, json

EMOTION_MAP = {
    1: "joy",
    2: "boredom",
    3: "hunger",
    4: "aggressivity",
    5: "sadness",
}

def read_wav_mono(path):
    """Read 16-bit PCM WAV, return float32 mono in [-1,1] and sample rate."""
    with wave.open(path, 'rb') as wf:
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth*8} bits. Please convert to 16-bit PCM.")
    if nchan == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    return data, fr

def frame_signal(x, sr, win_s=0.02, hop_s=0.01):
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    if win < 1: win = 1
    if hop < 1: hop = 1
    n = 1 + max(0, (len(x) - win) // hop)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, win),
        strides=(x.strides[0]*hop, x.strides[0]),
        writeable=False
    )
    return frames, win, hop

def rms_envelope(x, sr, win_s=0.02, hop_s=0.01):
    frames, win, hop = frame_signal(x, sr, win_s, hop_s)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    times = np.arange(len(rms)) * hop / sr
    return rms, times, win, hop

def find_segments(x, sr, min_len=0.05, max_len=0.6, win_s=0.02, hop_s=0.01, sens=1.5):
    """Energy-based segmentation using RMS threshold = median + sens*std."""
    rms, times, win, hop = rms_envelope(x, sr, win_s, hop_s)
    thr = np.median(rms) + sens*np.std(rms)
    mask = rms > thr
    segs = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start_i = i
            while i < len(mask) and mask[i]:
                i += 1
            end_i = i
            start = start_i * hop
            end = end_i * hop + win
            seg_len = (end - start) / sr
            if min_len <= seg_len <= max_len:
                segs.append((max(0, start), min(len(x), end)))
        i += 1
    # If no segments, fallback to whole clip (clipped to max_len if too long)
    if not segs:
        L = min(len(x), int(max_len*sr))
        segs = [(0, L)]
    return segs

def spectral_features(seg, sr):
    """Compute simple spectral features for one segment array (float32)."""
    n = len(seg)
    # Hann window
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/max(1,(n-1)))
    y = seg * w
    # FFT
    S = np.fft.rfft(y)
    mag = np.abs(S) + 1e-10
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    power = mag**2

    total_power = np.sum(power)
    if total_power <= 0: total_power = 1e-10

    # Spectral centroid
    centroid = np.sum(freqs * power) / total_power

    # Roll-off (0.85)
    cumsum = np.cumsum(power)
    idx = np.searchsorted(cumsum, 0.85 * total_power)
    rolloff = freqs[min(idx, len(freqs)-1)]

    # Zero-crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(seg))) > 0)

    # Spectral flatness (Wiener entropy)
    flatness = np.exp(np.mean(np.log(mag))) / (np.mean(mag))

    # Dominant frequency in 80-1200 Hz
    fmin, fmax = 80.0, 1200.0
    mask = (freqs >= fmin) & (freqs <= fmax)
    if np.any(mask):
        dom_idx = np.argmax(mag[mask])
        dom_freq = freqs[mask][dom_idx]
        dom_mag = mag[mask][dom_idx]
    else:
        dom_freq = 0.0
        dom_mag = 0.0

    # RMS energy
    rms = np.sqrt(np.mean(seg**2) + 1e-12)

    return np.array([centroid, rolloff, zcr, flatness, dom_freq, dom_mag, rms], dtype=np.float32)

def segment_features(x, sr):
    """Return a list of feature vectors for each detected bark segment in x."""
    segs = find_segments(x, sr)
    feats = []
    for s, e in segs:
        f = spectral_features(x[s:e], sr)
        feats.append(f)
    return np.vstack(feats), segs

def load_training(train_dir):
    """Load training WAVs 1.wav..5.wav and build a dataset by segmenting each into multiple samples."""
    X, y = [], []
    for idx in range(1, 6):
        path = os.path.join(train_dir, f"{idx}.wav")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training file missing: {path}")
        x, sr = read_wav_mono(path)
        feats, segs = segment_features(x, sr)
        for f in feats:
            X.append(f)
            y.append(idx)
    if not X:
        raise RuntimeError("No training segments extracted. Check your training WAVs.")
    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)
    return X, y

def knn_predict(X_train, y_train, x_query, k=5):
    """Simple k-NN (Euclidean). Returns predicted class and class probabilities."""
    if len(X_train) < k:
        k = max(1, len(X_train))
    dists = np.linalg.norm(X_train - x_query[None, :], axis=1)
    nn_idx = np.argpartition(dists, k-1)[:k]
    nn_labels = y_train[nn_idx]
    # majority vote
    classes = sorted(set(y_train.tolist()))
    counts = {c: int(np.sum(nn_labels == c)) for c in classes}
    pred = max(counts.items(), key=lambda kv: kv[1])[0]
    # probabilities (soft by 1/d)
    invd = 1.0 / (dists[nn_idx] + 1e-6)
    weights = invd / np.sum(invd)
    prob = {int(nn_labels[i]): float(weights[i]) for i in range(len(nn_labels))}
    # aggregate weights by class
    agg = {c: 0.0 for c in classes}
    for i, lbl in enumerate(nn_labels):
        agg[int(lbl)] += float(weights[i])
    # normalize
    s = sum(agg.values())
    if s <= 0: s = 1.0
    probs = {int(c): float(v/s) for c, v in agg.items()}
    return pred, probs, nn_idx.tolist()

def classify_clip(wav_path, train_dir, k=5):
    """Classify a new WAV by segmenting it and majority-voting over segment predictions."""
    Xtr, ytr = load_training(train_dir)
    x, sr = read_wav_mono(wav_path)
    Xq, segs = segment_features(x, sr)
    seg_preds = []
    seg_probs = []
    for i in range(Xq.shape[0]):
        pred, probs, _ = knn_predict(Xtr, ytr, Xq[i], k=k)
        seg_preds.append(pred)
        seg_probs.append(probs)
    # Majority vote over segments
    classes, counts = np.unique(seg_preds, return_counts=True)
    clip_pred = int(classes[np.argmax(counts)])
    # Average probabilities
    all_classes = sorted(set(ytr.tolist()))
    avg_probs = {int(c): 0.0 for c in all_classes}
    for p in seg_probs:
        for c in all_classes:
            avg_probs[int(c)] += float(p.get(int(c), 0.0))
    for c in avg_probs:
        avg_probs[c] /= max(len(seg_probs), 1)

    result = {
        "file": wav_path,
        "prediction_int": clip_pred,
        "prediction_label": EMOTION_MAP.get(clip_pred, str(clip_pred)),
        "segment_count": len(seg_preds),
        "segment_predictions": [int(p) for p in seg_preds],
        "average_class_probabilities": {EMOTION_MAP.get(k, str(k)): float(v) for k, v in avg_probs.items()},
    }
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_path", type=str, help="Path to WAV file to classify")
    ap.add_argument("--train-dir", type=str, default=None, help="Directory containing 1.wav..5.wav training files")
    ap.add_argument("--k", type=int, default=5, help="k for k-NN")
    args = ap.parse_args()

    if args.train_dir is None:
        args.train_dir = os.path.dirname(os.path.abspath(__file__))

    res = classify_clip(args.wav_path, args.train_dir, k=args.k)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
