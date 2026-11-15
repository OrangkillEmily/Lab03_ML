"""K-means tự cài đặt bằng NumPy (không sử dụng sklearn.cluster.KMeans)."""

import numpy as np


class KMeans:
    """K-means với Random/K-means++, lưu lịch sử và trạng thái hội tụ."""

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

        # Kết quả sau huấn luyện.
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None

        # Thông tin quá trình huấn luyện.
        self.n_iter_ = 0
        self.history_ = []
        self.converged_ = False
        self.stop_reason_ = None

    @staticmethod
    def _squared_distances(X, centers):
        """Tính bình phương khoảng cách Euclidean giữa các điểm và tâm."""

        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1)[None, :]

        distances = x2 + c2 - 2 * X @ centers.T

        # Tránh sai số số học tạo ra khoảng cách âm.
        return np.maximum(distances, 0.0)

    def _initialize_centers(self, X, rng):
        """Khởi tạo tâm cụm bằng Random hoặc K-means++."""

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

        # Chọn tâm đầu tiên ngẫu nhiên.
        centers = [
            X[rng.integers(n_samples)].copy()
        ]

        # Chọn các tâm tiếp theo dựa trên khoảng cách bình phương.
        for _ in range(1, self.n_clusters):

            distances = self._squared_distances(
                X,
                np.asarray(centers),
            )

            min_distances = distances.min(axis=1)
            total_distance = min_distances.sum()

            if total_distance <= 0:
                # Trường hợp tất cả điểm đều trùng nhau.
                candidate = X[rng.integers(n_samples)].copy()

            else:
                probabilities = (
                    min_distances / total_distance
                )

                candidate_idx = rng.choice(
                    n_samples,
                    p=probabilities,
                )

                candidate = X[candidate_idx].copy()

            centers.append(candidate)

        return np.asarray(centers, dtype=float)

    def _assign_labels(self, X, centers):
        """Gán mỗi điểm vào cụm có tâm gần nhất."""

        distances = self._squared_distances(
            X,
            centers,
        )

        labels = np.argmin(
            distances,
            axis=1,
        )

        return labels, distances

    def _update_centers(
        self,
        X,
        labels,
        centers,
        distances,
    ):
        """Cập nhật tâm cụm và xử lý các cụm rỗng."""

        new_centers = centers.copy()

        nearest_distances = distances.min(axis=1)

        for cluster_id in range(self.n_clusters):

            members = X[labels == cluster_id]

            if len(members) > 0:

                # Tâm mới bằng trung bình các điểm trong cụm.
                new_centers[cluster_id] = members.mean(
                    axis=0
                )

            else:

                # Chọn điểm xa tâm gần nhất để tái khởi tạo.
                farthest_idx = int(
                    np.argmax(nearest_distances)
                )

                new_centers[cluster_id] = X[
                    farthest_idx
                ]

                # Tránh chọn lại cùng điểm cho cụm rỗng tiếp theo.
                nearest_distances[farthest_idx] = -1.0

        return new_centers

    def _plot_iteration(self, X, labels, centers, iteration):
        """Vẽ trạng thái các cụm sau mỗi vòng lặp K-means."""

        import matplotlib.pyplot as plt
        from IPython.display import display

        if X.shape[1] != 2:
            raise ValueError(
                "plot_steps=True chỉ hỗ trợ dữ liệu 2D."
            )

        fig, ax = plt.subplots(figsize=(7, 5))

        # Vẽ các điểm dữ liệu theo cụm.
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=labels,
            cmap="tab10",
            s=15,
            alpha=0.5,
        )

        # Vẽ tâm cụm.
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="X",
            s=200,
            c="red",
            edgecolors="black",
            linewidths=1,
            label="Tâm cụm",
        )

        ax.set_title(
            f"K-means - Vòng lặp {iteration}"
        )
        ax.set_xlabel("Đặc trưng 1")
        ax.set_ylabel("Đặc trưng 2")
        ax.legend()
        ax.grid(alpha=0.25)

        plt.tight_layout()
        display(fig)
        plt.close(fig)

    def fit(self, X, plot_steps=False):
        """Huấn luyện mô hình K-means.

        Parameters
        ----------
        X : array-like
            Dữ liệu huấn luyện.
        plot_steps : bool
            Nếu True, trực quan hóa từng vòng lặp.
            Chỉ hỗ trợ dữ liệu có đúng 2 đặc trưng.
        """

        X = np.asarray(X, dtype=float)

        # 1. Kiểm tra tham số.
        if self.n_clusters < 1:
            raise ValueError(
                "n_clusters phải lớn hơn 0."
            )

        if self.max_iter < 1:
            raise ValueError(
                "max_iter phải lớn hơn 0."
            )

        if self.tol < 0:
            raise ValueError(
                "tol không được âm."
            )

        if self.init not in ("random", "k-means++"):
            raise ValueError(
                "init chỉ nhận 'random' hoặc 'k-means++'."
            )

        # 2. Kiểm tra dữ liệu.
        if X.ndim != 2:
            raise ValueError(
                "X phải là ma trận 2D."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "X không được rỗng."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X phải có ít nhất một đặc trưng."
            )

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "Số mẫu phải lớn hơn hoặc bằng số cụm."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X chứa NaN hoặc giá trị vô hạn."
            )

        if plot_steps and X.shape[1] != 2:
            raise ValueError(
                "plot_steps=True chỉ hỗ trợ dữ liệu 2D."
            )

        # 3. Reset trạng thái trước mỗi lần huấn luyện.
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None

        self.n_iter_ = 0
        self.history_ = []
        self.converged_ = False
        self.stop_reason_ = None

        # 4. Khởi tạo bộ sinh số ngẫu nhiên.
        rng = np.random.default_rng(
            self.random_state
        )

        # 5. Khởi tạo K tâm cụm.
        centers = self._initialize_centers(
            X,
            rng,
        )

        # 6. Vòng lặp K-means.
        for iteration in range(
            1,
            self.max_iter + 1,
        ):

            # Bước 1: Gán mỗi điểm vào cụm gần nhất.
            labels, distances = self._assign_labels(
                X,
                centers,
            )

            # Bước 2: Cập nhật tâm cụm.
            new_centers = self._update_centers(
                X,
                labels,
                centers,
                distances,
            )

            # Bước 3: Tính độ dịch chuyển của các tâm.
            centroid_shift = float(
                np.linalg.norm(
                    new_centers - centers
                )
            )

            centers = new_centers

            # Bước 4: Gán nhãn lại theo tâm mới.
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

            # Bước 6: Tính số lượng điểm trong mỗi cụm.
            cluster_sizes = np.bincount(
                labels,
                minlength=self.n_clusters,
            ).astype(int).tolist()

            # Bước 7: Lưu lịch sử huấn luyện.
            self.history_.append({
                "iteration": iteration,
                "inertia": inertia,
                "centroid_shift": centroid_shift,
                "cluster_sizes": cluster_sizes,
                "centers": centers.copy(),
            })

            # Bước 8: Vẽ từng vòng lặp nếu được yêu cầu.
            if plot_steps:
                self._plot_iteration(
                    X,
                    labels,
                    centers,
                    iteration,
                )

            # Bước 9: Kiểm tra điều kiện hội tụ.
            if centroid_shift <= self.tol:

                self.converged_ = True
                self.stop_reason_ = (
                    "centroid_shift <= tol"
                )

                break

        # 7. Ghi nhận trường hợp hết số vòng lặp.
        if not self.converged_:
            self.stop_reason_ = "max_iter reached"

        # 8. Lưu kết quả cuối cùng.
        self.centers_ = centers.copy()

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

    def fit_predict(self, X):
        """Huấn luyện mô hình và trả về nhãn cụm."""

        self.fit(X)

        return self.labels_.copy()
