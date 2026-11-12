import unittest
import numpy as np
from src.models.kmeans import KMeans
from src.experiments import internal_metrics


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [0., 1.], [9., 9.], [9., 8.], [10., 9.]])

    def test_both_initializers_fit(self):
        for init in ("random", "k-means++"):
            model = KMeans(n_clusters=2, init=init, random_state=7).fit(self.X)
            self.assertEqual(model.labels_.shape, (len(self.X),))
            self.assertEqual(model.centers_.shape, (2, 2))
            self.assertGreaterEqual(model.inertia_, 0)
            self.assertGreaterEqual(model.n_iter_, 1)

    def test_reproducible(self):
        a = KMeans(2, init="k-means++", random_state=10).fit(self.X)
        b = KMeans(2, init="k-means++", random_state=10).fit(self.X)
        np.testing.assert_allclose(a.centers_, b.centers_)
        np.testing.assert_array_equal(a.labels_, b.labels_)

    def test_predict_dimension_validation(self):
        model = KMeans(2).fit(self.X)
        with self.assertRaises(ValueError):
            model.predict(np.ones((2, 3)))

    def test_reject_non_finite(self):
        with self.assertRaises(ValueError):
            KMeans(2).fit(np.array([[0., 0.], [np.nan, 1.]]))

    def test_internal_metrics(self):
        model = KMeans(2, random_state=1).fit(self.X)
        ch, db = internal_metrics(self.X, model.labels_, model.centers_)
        self.assertGreater(ch, 0)
        self.assertGreaterEqual(db, 0)


if __name__ == "__main__":
    unittest.main()
