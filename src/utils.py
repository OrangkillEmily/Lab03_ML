"""Các hàm tiện ích dùng chung trong dự án."""
import numpy as np


def silhouette_score_sample(X, labels, sample_size=1000, random_state=42):
    """Ước lượng Silhouette trên mẫu bằng ma trận khoảng cách Euclidean.

    Hàm vector hóa theo cụm để nhanh hơn cách lặp từng điểm; độ phức tạp bộ nhớ
    O(sample_size^2), vì vậy nên giữ sample_size khoảng 500-1.500.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if len(X) != len(labels):
        raise ValueError("X và labels phải có cùng số phần tử.")
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X, labels = X[idx], labels[idx]

    groups = np.unique(labels)
    n = len(X)
    if len(groups) < 2 or len(groups) >= n:
        return float("nan")

    squared = np.sum(X * X, axis=1)
    distances = np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2 * X @ X.T, 0.0))
    mean_distances = np.empty((n, len(groups)), dtype=float)
    own_counts = np.empty(len(groups), dtype=int)
    own_masks = []
    for j, group in enumerate(groups):
        mask = labels == group
        count = int(mask.sum())
        own_counts[j] = count
        own_masks.append(mask)
        mean_distances[:, j] = distances[:, mask].mean(axis=1)

    scores = np.zeros(n, dtype=float)
    for j, mask in enumerate(own_masks):
        # Bỏ chính điểm đó khỏi khoảng cách trung bình nội cụm.
        if own_counts[j] <= 1:
            scores[mask] = 0.0
            continue
        a = (distances[np.ix_(mask, mask)].sum(axis=1)) / (own_counts[j] - 1)
        other_means = mean_distances[mask].copy()
        other_means[:, j] = np.inf
        b = other_means.min(axis=1)
        denom = np.maximum(a, b)
        scores[mask] = np.divide(b - a, denom, out=np.zeros_like(a), where=denom > 0)
    return float(scores.mean())
