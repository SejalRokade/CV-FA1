import argparse
from typing import Tuple

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_bloodmnist
from src.preprocess import to_grayscale_enhanced, denoise
from src.segment import otsu_segment, largest_component
from src.features import (
    compute_shape_features, compute_texture_features,
    compute_local_descriptors, build_bovw_codebook, descriptors_to_bovw
)


def augment_image(image: np.ndarray) -> np.ndarray:
    """Simple data augmentation for blood cell images."""
    augmented = []
    
    # Original image
    augmented.append(image)
    
    # Horizontal flip
    augmented.append(cv2.flip(image, 1))
    
    # Small rotations
    for angle in [-15, 15]:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        augmented.append(rotated)
    
    # Small brightness/contrast adjustments
    for alpha in [0.9, 1.1]:  # contrast
        for beta in [-10, 10]:  # brightness
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            augmented.append(adjusted)
    
    return augmented


def extract_features(images: np.ndarray, feature_method: str, codebook=None, augment: bool = False):
    """Extract features with optional augmentation."""
    X_shape = []
    X_texture = []
    local_descs = []
    
    for img in images:
        if augment:
            # Apply augmentation
            augmented_imgs = augment_image(img)
            # Use the first augmented image for feature extraction
            img = augmented_imgs[0]
        
        # Preprocess
        gray = to_grayscale_enhanced(img)
        gray = denoise(gray, ksize=5)
        
        # Segment
        mask = otsu_segment(gray)
        mask = largest_component(mask)
        
        # Shape features
        shape_feat = compute_shape_features(mask)
        X_shape.append(shape_feat)
        
        # Texture features
        texture_feat = compute_texture_features(gray, mask)
        X_texture.append(texture_feat)
        
        # Local descriptors only if not "none"
        if feature_method != "none":
            masked = cv2.bitwise_and(gray, gray, mask=mask)
            desc = compute_local_descriptors(masked, method=feature_method)
            local_descs.append(desc)
    
    X_shape = np.stack(X_shape, axis=0)
    X_texture = np.stack(X_texture, axis=0)
    
    if feature_method == "none":
        return np.concatenate([X_shape, X_texture], axis=1), None
    
    if codebook is None:
        codebook = build_bovw_codebook(local_descs, k=128)
    
    bovw_features = [descriptors_to_bovw(d, codebook) for d in local_descs]
    X_bovw = np.stack(bovw_features, axis=0)
    
    # Combine all features
    X = np.concatenate([X_shape, X_texture, X_bovw], axis=1)
    return X, codebook


def get_best_classifier(X_train: np.ndarray, y_train: np.ndarray, clf_type: str = "auto"):
    """Get the best classifier with hyperparameter tuning."""
    
    if clf_type == "auto":
        # Try multiple classifiers and pick the best
        classifiers = ["svm", "rf", "gb", "mlp"]
        best_score = 0
        best_clf = None
        
        for clf_name in classifiers:
            print(f"Tuning {clf_name.upper()}...")
            clf, score = get_best_classifier(X_train, y_train, clf_name)
            if score > best_score:
                best_score = score
                best_clf = clf
            print(f"{clf_name.upper()} CV score: {score:.4f}")
        
        return best_clf, best_score
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', None)
    ])
    
    if clf_type == "svm":
        # SVM with RBF kernel
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'classifier__class_weight': ['balanced']
        }
        base_clf = SVC(kernel='rbf', random_state=42, probability=True)
        
    elif clf_type == "rf":
        # Random Forest
        param_grid = {
            'classifier__n_estimators': [200, 300, 500],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': ['balanced']
        }
        base_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
    elif clf_type == "gb":
        # Gradient Boosting
        param_grid = {
            'classifier__n_estimators': [200, 300, 500],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
        base_clf = GradientBoostingClassifier(random_state=42)
        
    elif clf_type == "mlp":
        # Neural Network
        param_grid = {
            'classifier__hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate_init': [0.001, 0.01, 0.1]
        }
        base_clf = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
        
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")
    
    pipeline.set_params(classifier=base_clf)
    
    # Cross-validation with stratified splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', 
        n_jobs=-1, verbose=1, error_score=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_score_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bloodmnist")
    parser.add_argument("--feature", choices=["sift", "orb", "none"], default="sift")
    parser.add_argument("--clf", choices=["auto", "svm", "rf", "gb", "mlp"], default="auto")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--model-out", type=str, default=None)
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--fast", action="store_true", help="Use smaller dataset for fast training")
    args = parser.parse_args()

    if args.dataset != "bloodmnist":
        raise ValueError("Only bloodmnist is supported in this script")

    print("Loading BloodMNIST dataset...")
    data = load_bloodmnist(val_split=args.val_split)
    
    # Use smaller subset for fast training
    if args.fast:
        print("Fast mode: Using smaller dataset subset...")
        # Use only 20% of data for fast training
        train_size = int(0.2 * len(data.x_train))
        val_size = int(0.2 * len(data.x_val))
        test_size = int(0.2 * len(data.x_test))
        
        data.x_train = data.x_train[:train_size]
        data.y_train = data.y_train[:train_size]
        data.x_val = data.x_val[:val_size]
        data.y_val = data.y_val[:val_size]
        data.x_test = data.x_test[:test_size]
        data.y_test = data.y_test[:test_size]
    
    print(f"Dataset sizes:")
    print(f"  Train: {data.x_train.shape[0]}")
    print(f"  Val: {data.x_val.shape[0]}")
    print(f"  Test: {data.x_test.shape[0]}")

    print(f"\nExtracting features (method: {args.feature}, augment: {args.augment})...")
    X_train, codebook = extract_features(data.x_train, args.feature, augment=args.augment)
    X_val, _ = extract_features(data.x_val, args.feature, codebook=codebook)
    X_test, _ = extract_features(data.x_test, args.feature, codebook=codebook)
    
    print(f"Feature dimensions:")
    print(f"  Shape: {X_train.shape[1]}")
    if codebook is not None:
        print(f"  Codebook: {codebook.shape[0]}")
    else:
        print(f"  No local descriptors (shape + texture only)")

    # Combine train and val for final training
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([data.y_train, data.y_val], axis=0)
    
    print(f"\nTraining classifier ({args.clf})...")
    clf, cv_score = get_best_classifier(X_combined, y_combined, args.clf)
    
    print(f"\nFinal training on combined dataset...")
    clf.fit(X_combined, y_combined)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(data.y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  CV Score: {cv_score:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(data.y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(data.y_test, y_pred)
    print(cm)

    # Save model by default
    model_out = args.model_out or "blood_cell_model.pkl"
    joblib.dump({"clf": clf, "codebook": codebook}, model_out)
    print(f"\nSaved model to {model_out}")


if __name__ == "__main__":
    main()