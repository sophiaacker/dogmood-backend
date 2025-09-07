#!/usr/bin/env python3
# kNN classifier for dog skin issues (dataset: dog_skin_classifier/{ear,atopic,acute,lick})
# Features: HSV histograms + uniform LBP (robust for tiny datasets ~20 images)

import argparse
import os
import glob
from collections import Counter
import numpy as np
from PIL import Image
import cv2

from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.exceptions import NotFittedError
import joblib

# The exact expected subfolders (as provided by the user)
EXPECTED_LABELS = ["ear", "atopic", "acute", "lick"]
IMG_SIZE = 256  # resize target
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def verify_structure(data_dir):
    """Ensure data_dir contains the expected four subfolders. Print helpful tips."""
    missing = []
    found = []
    for lab in EXPECTED_LABELS:
        p = os.path.join(data_dir, lab)
        if os.path.isdir(p):
            found.append(lab)
        else:
            missing.append(lab)
    print("Dataset root:", data_dir)
    print("Expected subfolders:", ", ".join(EXPECTED_LABELS))
    if missing:
        print("[Warning] Missing subfolders:", ", ".join(missing))
    if found:
        print("[OK] Found subfolders:", ", ".join(found))

def list_images_with_labels(data_dir):
    """Collect image paths and labels from expected subfolders only (ear, atopic, acute, lick)."""
    X_paths, y = [], []
    for cls in EXPECTED_LABELS:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for p in sorted(glob.glob(os.path.join(cls_dir, "**", "*"), recursive=True)):
            if os.path.splitext(p)[1].lower() in VALID_EXTS:
                X_paths.append(p)
                y.append(cls)
    if not X_paths:
        raise ValueError(f"No images found under {data_dir}. "
                         f"Make sure you have: {', '.join(EXPECTED_LABELS)} with images inside.")
    return X_paths, np.array(y)

def load_image_bgr(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        im = im.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(im)[:, :, ::-1]  # RGB -> BGR
    return arr

def extract_features(img_bgr):
    """HSV hist (16/8/8 bins) + LBP uniform (P=8,R=1 -> 10 bins)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    def norm_hist(channel, bins, rmax):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, rmax]).flatten().astype("float32")
        s = hist.sum()
        return hist / s if s > 0 else hist

    h_hist = norm_hist(h, 16, 180)
    s_hist = norm_hist(s, 8, 256)
    v_hist = norm_hist(v, 8, 256)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    P, R = 8, 1
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    lbp_bins = np.arange(0, P + 3)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, P + 2))
    lbp_hist = lbp_hist.astype("float32")
    ssum = lbp_hist.sum()
    lbp_hist = lbp_hist / ssum if ssum > 0 else lbp_hist

    return np.concatenate([h_hist, s_hist, v_hist, lbp_hist]).astype("float32")

def build_feature_matrix(paths):
    X = []
    for p in paths:
        img = load_image_bgr(p)
        X.append(extract_features(img))
    return np.vstack(X)

def select_k_via_cv(X, y, max_k=9):
    counts = Counter(y)
    min_class = min(counts.values())
    n = len(y)
    ks = [k for k in range(1, min(max_k, n - 1) + 1) if k % 2 == 1]
    if not ks:
        return 1
    if min_class < 2 or n < 4:
        return min(3, max(ks))
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(weights="distance", metric="euclidean"))
    ])
    gs = GridSearchCV(pipe, {"knn__n_neighbors": ks}, cv=cv, n_jobs=-1, scoring="accuracy")
    gs.fit(X, y)
    print(f"[CV] best k={gs.best_params_['knn__n_neighbors']}, mean acc={gs.best_score_:.3f} (cv={n_splits}-fold)")
    return int(gs.best_params_["knn__n_neighbors"])

def print_banner_and_counts(y):
    print("\n=== Labels (fixed order) ===")
    print(" -> " + " | ".join(EXPECTED_LABELS))
    counts = Counter(y)
    print("\n[Class counts]")
    for lab in EXPECTED_LABELS:
        print(f"  {lab}: {counts.get(lab, 0)}")

def train(data_dir, model_out):
    verify_structure(data_dir)
    paths, y = list_images_with_labels(data_dir)
    print(f"Found {len(paths)} images total.")
    print_banner_and_counts(y)

    X = build_feature_matrix(paths)
    k = select_k_via_cv(X, y, max_k=9)
    print(f"Using k={k}")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean"))
    ])
    clf.fit(X, y)

    # Sanity report (on training set)
    y_pred = clf.predict(X)
    print("\n[Training-set sanity check report] (optimistic)")
    print(classification_report(y, y_pred, labels=EXPECTED_LABELS, target_names=EXPECTED_LABELS, zero_division=0))

    joblib.dump({"pipeline": clf, "labels": EXPECTED_LABELS, "feature_version": "hsv16-8-8+lbp-u8-r1"}, model_out)
    print(f"\nSaved model to: {model_out}")

def predict(model_path, image_path, topk=3):
    payload = joblib.load(model_path)
    clf = payload["pipeline"]

    img = load_image_bgr(image_path)
    x = extract_features(img).reshape(1, -1)

    try:
        proba = clf.predict_proba(x)[0]
        classes = clf.classes_
        pred_idx = int(np.argmax(proba))
        pred_label = classes[pred_idx]
        print("\n=== Prediction ===")
        print(f"Predicted label: {pred_label}")
        order = np.argsort(proba)[::-1][:max(1, topk)]
        print("\nTop probabilities:")
        for i in order:
            print(f"  {classes[i]}: {proba[i]:.3f}")
    except (AttributeError, NotFittedError):
        pred_label = clf.predict(x)[0]
        print("\n=== Prediction ===")
        print(f"Predicted label: {pred_label}")

def main():
    ap = argparse.ArgumentParser(description="kNN for dog skin images (ear, atopic, acute, lick).")
    ap.add_argument("--data-dir", default="dog_skin_classifier", help="Root dataset folder (default: dog_skin_classifier)")
    ap.add_argument("--model-out", default="knn_skin.joblib", help="Output model path (default: knn_skin.joblib)")
    ap.add_argument("--image", help="For prediction mode: path to image to classify")
    ap.add_argument("--topk", type=int, default=3, help="Show top-k probabilities in prediction")
    ap.add_argument("--mode", choices=["train", "predict"], required=True, help="Run 'train' or 'predict'")

    args = ap.parse_args()

    if args.mode == "train":
        train(args.data_dir, args.model_out)
    else:
        if not args.image:
            raise SystemExit("--image is required in predict mode")
        predict(args.model_out, args.image, args.topk)

if __name__ == "__main__":
    main()
