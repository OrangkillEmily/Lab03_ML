"""A from-scratch K-Means implementation with K-Means++ initialization."""

import numpy as np


class KMeansFromScratch:
    """
    K-Means clustering for RGB pixel data.

    Uses K-Means++ initialization and Euclidean distance.
    """

    def __init__(
        self,
        n_clusters=3,
        max_iter=100,
        tol=1e-3,
        random_state=42,
        batch_size=50000,
    ):
        if (
            isinstance(n_clusters, bool)
            or not isinstance(n_clusters, (int, np.integer))
            or n_clusters < 1
        ):
            raise ValueError("n_clusters phải là số nguyên >= 1.")

        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, (int, np.integer))
            or max_iter < 1
        ):
            raise ValueError("max_iter phải là số nguyên >= 1.")

        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol phải là số hữu hạn >= 0.")

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size phải là số nguyên >= 1.")

        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.batch_size = int(batch_size)

        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.labels_ = None

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def _validate_X(X):
        """Validate RGB feature matrix."""

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X phải có dạng Nx3 RGB.")

        if len(X) == 0:
            raise ValueError("X không được rỗng.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X chứa NaN hoặc Infinity.")

        if np.any(X < 0) or np.any(X > 255):
            raise ValueError(
                "Giá trị RGB phải nằm trong khoảng [0, 255]."
            )

        return X

    # --------------------------------------------------------
    # DISTANCE COMPUTATION
    # --------------------------------------------------------

    @staticmethod
    def _squared_distances(X, centers):
        """Calculate squared Euclidean distances."""

        return np.sum(
            (X[:, None, :] - centers[None, :, :]) ** 2,
            axis=2
        )

    def _assign_labels(self, X, centers):
        """Assign samples to nearest centers using batches."""

        labels = np.empty(
            len(X),
            dtype=np.int32
        )

        nearest_distances = np.empty(
            len(X),
            dtype=np.float32
        )

        for start in range(0, len(X), self.batch_size):

            end = min(
                start + self.batch_size,
                len(X)
            )

            batch = X[start:end]

            distances = self._squared_distances(
                batch,
                centers
            )

            labels[start:end] = np.argmin(
                distances,
                axis=1
            )

            nearest_distances[start:end] = np.min(
                distances,
                axis=1
            )

        return labels, nearest_distances

    # --------------------------------------------------------
    # K-MEANS++ INITIALIZATION
    # --------------------------------------------------------

    def _init_centers(self, X, rng):
        """Initialize cluster centers using K-Means++."""

        n_samples = len(X)

        centers = np.empty(
            (self.n_clusters, X.shape[1]),
            dtype=np.float32
        )

        first_idx = rng.integers(n_samples)

        centers[0] = X[first_idx]

        closest_distances = np.sum(
            (X - centers[0]) ** 2,
            axis=1
        )

        for center_idx in range(1, self.n_clusters):

            total_distance = closest_distances.sum()

            if total_distance <= 0:
                # Khi tất cả điểm đã chọn có khoảng cách bằng 0,
                # chọn ngẫu nhiên một điểm còn lại.
                selected_idx = rng.integers(n_samples)

            else:
                probabilities = (
                    closest_distances / total_distance
                )

                selected_idx = rng.choice(
                    n_samples,
                    p=probabilities
                )

            centers[center_idx] = X[selected_idx]

            new_distances = np.sum(
                (X - centers[center_idx]) ** 2,
                axis=1
            )

            closest_distances = np.minimum(
                closest_distances,
                new_distances
            )

        return centers

    # --------------------------------------------------------
    # EMPTY CLUSTER HANDLING
    # --------------------------------------------------------

    def _update_centers(self, X, labels, nearest_distances, centers):
        """Update centers and reinitialize empty clusters."""

        new_centers = centers.copy()

        cluster_counts = np.bincount(
            labels,
            minlength=self.n_clusters
        )

        empty_clusters = np.where(
            cluster_counts == 0
        )[0]

        # Tính trung bình cho các cụm không rỗng.
        for cluster_idx in range(self.n_clusters):

            if cluster_counts[cluster_idx] > 0:

                members = X[
                    labels == cluster_idx
                ]

                new_centers[cluster_idx] = members.mean(
                    axis=0
                )

        # Khởi tạo lại từng cụm rỗng bằng điểm có sai số lớn.
        if len(empty_clusters) > 0:

            candidate_distances = nearest_distances.copy()

            for cluster_idx in empty_clusters:

                farthest_idx = int(
                    np.argmax(candidate_distances)
                )

                new_centers[cluster_idx] = X[
                    farthest_idx
                ]

                # Tránh chọn lại đúng điểm vừa sử dụng.
                candidate_distances[farthest_idx] = -1

        return new_centers

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(self, X):
        """Fit K-Means model to RGB samples."""

        X = self._validate_X(X)

        if len(X) < self.n_clusters:
            raise ValueError(
                "Số mẫu phải lớn hơn hoặc bằng n_clusters."
            )

        rng = np.random.default_rng(
            self.random_state
        )

        centers = self._init_centers(
            X,
            rng
        )

        self.n_iter_ = 0

        for iteration in range(self.max_iter):

            labels, nearest_distances = self._assign_labels(
                X,
                centers
            )

            new_centers = self._update_centers(
                X,
                labels,
                nearest_distances,
                centers
            )

            shift = np.linalg.norm(
                new_centers - centers
            )

            centers = new_centers

            self.n_iter_ = iteration + 1

            if shift <= self.tol:
                break

        # Tính lại labels sau khi cập nhật tâm cụm cuối cùng.
        final_labels, final_distances = self._assign_labels(
            X,
            centers
        )

        self.cluster_centers_ = centers
        self.labels_ = final_labels

        self.inertia_ = float(
            np.sum(final_distances, dtype=np.float64)
        )

        return self

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(self, X, batch_size=None):
        """Predict nearest cluster for each RGB sample."""

        if self.cluster_centers_ is None:
            raise RuntimeError(
                "Cần fit mô hình trước khi predict."
            )

        X = self._validate_X(X)

        if batch_size is None:
            batch_size = self.batch_size

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError(
                "batch_size phải là số nguyên >= 1."
            )

        original_batch_size = self.batch_size

        try:
            self.batch_size = int(batch_size)

            labels, _ = self._assign_labels(
                X,
                self.cluster_centers_
            )

        finally:
            self.batch_size = original_batch_size

        return labels

    # --------------------------------------------------------
    # FIT PREDICT
    # --------------------------------------------------------

    def fit_predict(self, X):
        """Fit the model and return cluster labels."""

        return self.fit(X).predict(X)


# ============================================================
# RANDOM SAMPLING
# ============================================================

def make_sample(
    X,
    max_samples=20000,
    random_state=42
):
    """
    Randomly sample rows from X without replacement.

    Parameters
    ----------
    X : np.ndarray
        Input pixel matrix.
    max_samples : int
        Maximum number of samples.
    random_state : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Sampled pixel matrix.
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            "X phải là ma trận hai chiều."
        )

    if len(X) == 0:
        raise ValueError(
            "Không thể lấy mẫu từ dữ liệu rỗng."
        )

    if (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, (int, np.integer))
        or max_samples < 1
    ):
        raise ValueError(
            "max_samples phải là số nguyên >= 1."
        )

    if len(X) <= max_samples:
        return X.copy()

    rng = np.random.default_rng(
        random_state
    )

    indices = rng.choice(
        len(X),
        size=max_samples,
        replace=False
    )

    return X[indices]
