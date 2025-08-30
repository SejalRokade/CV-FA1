from typing import List, Tuple, Optional

import cv2
import numpy as np
from skimage.measure import moments_hu, regionprops, label
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor


def compute_shape_features(mask: np.ndarray) -> np.ndarray:
    """Compute comprehensive shape features from binary mask."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255

    # Hu moments
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    # Log transform for scale invariance
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    # Region properties
    lbl = label(mask > 0)
    features = []
    
    if lbl.max() > 0:
        props = regionprops(lbl)
        # choose largest area component
        props.sort(key=lambda p: p.area, reverse=True)
        prop = props[0]
        
        # Basic shape features
        features.extend([
            float(prop.area),  # Area
            float(prop.perimeter),  # Perimeter
            float(prop.eccentricity),  # Eccentricity
            float(prop.solidity),  # Solidity
            float(prop.extent),  # Extent
            float(prop.major_axis_length),  # Major axis length
            float(prop.minor_axis_length),  # Minor axis length
            float(prop.orientation),  # Orientation
            float(prop.equivalent_diameter),  # Equivalent diameter
            float(prop.feret_diameter_max),  # Maximum Feret diameter
        ])
        
        # Additional shape features
        if prop.area > 0:
            features.extend([
                float(prop.perimeter / np.sqrt(prop.area)),  # Compactness
                float(prop.area / (prop.major_axis_length * prop.minor_axis_length)),  # Rectangularity
                float(4 * np.pi * prop.area / (prop.perimeter ** 2)),  # Circularity
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0] * 13)  # Default values if no region found

    # Combine all features
    all_features = np.concatenate([hu, np.array(features, dtype=np.float32)], axis=0)
    
    # Normalize features (robust scaling)
    all_features = robust_normalize(all_features)
    
    return all_features


def compute_texture_features(image_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute texture features using GLCM and LBP."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Apply mask
    masked_image = image_gray.copy()
    masked_image[mask == 0] = 0
    
    # GLCM features
    glcm_features = []
    distances = [1, 2, 3]
    angles = [0, 45, 90, 135]
    
    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(masked_image, [distance], [np.radians(angle)], 
                               levels=256, symmetric=True, normed=True)
            
            # GLCM properties
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            
            glcm_features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
    
    # LBP features
    lbp = local_binary_pattern(masked_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    
    # Gabor filter features
    gabor_features = []
    frequencies = [0.1, 0.3, 0.5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for frequency in frequencies:
        for angle in angles:
            filt_real, filt_imag = gabor(masked_image, frequency=frequency, theta=angle)
            gabor_features.extend([
                np.mean(filt_real),
                np.std(filt_real),
                np.mean(filt_imag),
                np.std(filt_imag)
            ])
    
    # Combine all texture features
    texture_features = np.concatenate([
        np.array(glcm_features, dtype=np.float32),
        lbp_hist.astype(np.float32),
        np.array(gabor_features, dtype=np.float32)
    ])
    
    # Normalize
    texture_features = robust_normalize(texture_features)
    
    return texture_features


def robust_normalize(features: np.ndarray) -> np.ndarray:
    """Robust normalization using median and MAD."""
    median = np.median(features)
    mad = np.median(np.abs(features - median))
    
    if mad > 0:
        normalized = (features - median) / (1.4826 * mad)  # 1.4826 is consistency factor
    else:
        normalized = features - median
    
    # Clip outliers
    normalized = np.clip(normalized, -3, 3)
    
    return normalized.astype(np.float32)


def _create_feature_detector(name: str):
    name = name.lower()
    if name == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10)
    if name in ("orb", "sift"):
        return cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
    raise ValueError("Unsupported feature type: %s" % name)


def compute_local_descriptors(image_gray: np.ndarray, method: str = "sift") -> np.ndarray:
    detector = _create_feature_detector(method)
    keypoints, descriptors = detector.detectAndCompute(image_gray, None)
    if descriptors is None or len(keypoints) == 0:
        return np.zeros((0, detector.descriptorSize()), dtype=np.float32)
    return descriptors.astype(np.float32)


def build_bovw_codebook(descriptors_list: List[np.ndarray], k: int = 128, seed: int = 42) -> np.ndarray:
    """Enhanced KMeans codebook with better initialization."""
    from sklearn.cluster import MiniBatchKMeans
    
    all_desc = np.concatenate([d for d in descriptors_list if d.size > 0], axis=0)
    if all_desc.shape[0] < k:
        k = max(2, min(k, all_desc.shape[0]))
    
    # Better initialization and parameters
    kmeans = MiniBatchKMeans(
        n_clusters=k, 
        random_state=seed, 
        batch_size=1024,
        max_iter=300,
        init='k-means++',
        n_init=3,
        reassignment_ratio=0.01
    )
    kmeans.fit(all_desc)
    return kmeans.cluster_centers_.astype(np.float32)


def descriptors_to_bovw(descriptors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    from sklearn.metrics import pairwise_distances_argmin_min

    if descriptors.size == 0:
        return np.zeros((codebook.shape[0],), dtype=np.float32)
    
    # Soft assignment for better representation
    distances = np.linalg.norm(descriptors[:, np.newaxis, :] - codebook[np.newaxis, :, :], axis=2)
    
    # Soft assignment with temperature
    temperature = 0.1
    soft_assignments = np.exp(-distances / temperature)
    soft_assignments /= np.sum(soft_assignments, axis=1, keepdims=True)
    
    # Aggregate
    hist = np.sum(soft_assignments, axis=0).astype(np.float32)
    
    # Normalize
    hist /= (np.linalg.norm(hist) + 1e-8)
    
    return hist


