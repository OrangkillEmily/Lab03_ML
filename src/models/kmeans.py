"""K-means tự cài đặt bằng NumPy (không gọi sklearn.cluster.KMeans)."""

import numpy as np


class KMeans:
    """K-means với khởi tạo random/K-means++, lưu lịch sử hội tụ.

    Parameters
    ----------
    n_clusters : int
        Số cụm K.
    max_iter : int
        Số vòng lặp tối đa.
    init : {'k-means++', 'random'}
        Cách chọn tâm ban đầu.
    tol : float
        Ngưỡng dừng dựa trên độ dịch chuyển của các tâm.
    random_state : int | None
        Seed để tái lập kết quả.
    """

    def __init__(
        self,
        n_clusters=3,
        max_iter=100,
        init="k-means++",
        tol=1e-4,
        random_state=42,
    ):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.init = init.lower().replace("_", "-")
        self.tol = float(tol)
        self.random_state = random_state

        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.history_ = []

    @staticmethod
    def _squared_distances(X, centers):
        """Tính bình phương khoảng cách Euclidean giữa các điểm và tâm cụm."""
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1)[None, :]

        distances = x2 + c2 - 2 * X @ centers.T

        return np.maximum(distances, 0.0)

    def _initialize_centers(self, X, rng):
        """Khởi tạo tâm cụm bằng random hoặc K-means++."""
        n_samples = len(X)

        if self.init == "random":
            indices = rng.choice(
                n_samples,
                size=self.n_clusters,
                replace=False,
            )

            return X[indices].copy()

        if self.init != "k-means++":
            raise ValueError(
                "init chỉ nhận 'random' hoặc 'k-means++'."
            )

        centers = [
            X[rng.integers(n_samples)].copy()
        ]

        for _ in range(1, self.n_clusters):
            distances = self._squared_distances(
                X,
                np.asarray(centers),
            )

            min_distances = distances.min(axis=1)
            total_distance = min_distances.sum()

            if total_distance <= 0:
                # Trường hợp dữ liệu có nhiều điểm trùng nhau.
                candidate = X[rng.integers(n_samples)].copy()
            else:
                probabilities = min_distances / total_distance

                candidate_idx = rng.choice(
                    n_samples,
                    p=probabilities,
                )

                candidate = X[candidate_idx].copy()

            centers.append(candidate)

        return np.asarray(centers, dtype=float)

    def _assign_labels(self, X, centers):
        """Gán mỗi điểm vào cụm có tâm gần nhất."""
        distances = self._squared_distances(X, centers)

        labels = np.argmin(distances, axis=1)

        return labels, distances

    def _update_centers(self, X, labels, centers, distances):
        """Cập nhật tâm cụm và xử lý cụm rỗng."""
        new_centers = centers.copy()

        nearest_distances = distances.min(axis=1)

        for cluster_id in range(self.n_clusters):
            members = X[labels == cluster_id]

            if len(members) > 0:
                # Tâm mới là trung bình các điểm trong cụm.
                new_centers[cluster_id] = members.mean(axis=0)

            else:
                # Chọn điểm xa tâm gần nhất để tái khởi tạo cụm rỗng.
                farthest_idx = int(
                    np.argmax(nearest_distances)
                )

                new_centers[cluster_id] = X[farthest_idx]

                # Không chọn lại cùng một điểm cho cụm rỗng tiếp theo.
                nearest_distances[farthest_idx] = -1.0

        return new_centers

    def fit(self, X):
        """Huấn luyện K-means."""
        X = np.asarray(X, dtype=float)

        # Kiểm tra dữ liệu.
        if X.ndim != 2:
            raise ValueError("X phải là ma trận 2D.")

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "Số mẫu phải lớn hơn hoặc bằng số cụm."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X phải có ít nhất một đặc trưng."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X chứa NaN hoặc giá trị vô hạn."
            )

        if self.n_clusters < 1:
            raise ValueError("n_clusters phải lớn hơn 0.")

        if self.max_iter < 1:
            raise ValueError("max_iter phải lớn hơn 0.")

        if self.tol < 0:
            raise ValueError("tol không được âm.")

        # Khởi tạo.
        rng = np.random.default_rng(self.random_state)

        centers = self._initialize_centers(X, rng)

        self.history_ = []

        # Vòng lặp K-means.
        for iteration in range(1, self.max_iter + 1):

            # Bước 1: Gán nhãn.
            labels, distances = self._assign_labels(
                X,
                centers,
            )

            # Bước 2: Cập nhật tâm.
            new_centers = self._update_centers(
                X,
                labels,
                centers,
                distances,
            )

            # Bước 3: Tính độ dịch chuyển tâm.
            centroid_shift = float(
                np.linalg.norm(new_centers - centers)
            )

            centers = new_centers

            # Bước 4: Gán nhãn theo tâm mới.
            labels, distances = self._assign_labels(
                X,
                centers,
            )

            # Bước 5: Tính Inertia.
            selected_distances = distances[
                np.arange(len(X)),
                labels,
            ]

            inertia = float(
                selected_distances.sum()
            )

            # Bước 6: Kích thước các cụm.
            cluster_sizes = np.bincount(
                labels,
                minlength=self.n_clusters,
            ).astype(int).tolist()

            # Lưu lịch sử hội tụ.
            self.history_.append({
                "iteration": iteration,
                "inertia": inertia,
                "centroid_shift": centroid_shift,
                "cluster_sizes": cluster_sizes,
                "centers": centers.copy(),
            })

            # Điều kiện dừng.
            if centroid_shift <= self.tol:
                break

        # Lưu kết quả cuối cùng.
        self.centers_ = centers

        self.labels_ = self._assign_labels(
            X,
            self.centers_,
        )[0]

        final_distances = self._squared_distances(
            X,
            self.centers_,
        )

        self.inertia_ = float(
            final_distances[
                np.arange(len(X)),
                self.labels_,
            ].sum()
        )

        self.n_iter_ = iteration

        return self

    @property
    def cluster_centers_(self):
        """Tên thuộc tính tương thích với quy ước sklearn."""
        return self.centers_

    def predict(self, X):
        """Dự đoán cụm cho dữ liệu mới."""
        if self.centers_ is None:
            raise ValueError(
                "Cần gọi fit() trước predict()."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X phải là ma trận 2D."
            )

        if X.shape[1] != self.centers_.shape[1]:
            raise ValueError(
                "X mới phải có cùng số đặc trưng "
                "với dữ liệu huấn luyện."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X chứa NaN hoặc giá trị vô hạn."
            )

        return self._assign_labels(
            X,
            self.centers_,
        )[0]
