"""Tests for the knn_regression function"""

import unittest
import numpy as np

from src.knn import knn_regression

class TestKnn(unittest.TestCase):
    """Test class for knn regression function"""
    @classmethod
    def test_smoke(cls):
        """Simple smoke test to make sure function runs."""
        knn_regression(3, [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495],
                       [2, 5, 260]], [5, 4])

    def test_n_neighbors_is_zero(self):
        """Edge test to make sure the function throws a ValueError when the n_neighbors is zero."""
        with self.assertRaises(ValueError):
            knn_regression(0, [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495],
                           [2, 5, 260]], [5, 4])

    def test_n_neighbors_is_greater_than_the_sample_size_of_data(self):
        """Edge test to make sure the function throws a ValueError when the n-neighbors is greater
           than the sample size of data."""
        with self.assertRaises(ValueError):
            knn_regression(6, [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495],
                           [2, 5, 260]], [5, 4])

    def test_n_neighbors_is_empty(self):
        """test to make sure the function throws a TypeError when the n_neighbors is empty."""
        with self.assertRaises(TypeError):
            knn_regression(n_neighbors = [], data = [[3, 1, 230], [6, 2, 745], [6, 6, 1080],
                           [4, 3, 495], [2, 5, 260]], query = [5, 4])

    def test_query_is_empty(self):
        """test to make sure the function throws a TypeError when the query is empty."""
        with self.assertRaises(TypeError):
            knn_regression(n_neighbors = 3, data = [[3, 1, 230], [6, 2, 745],
                          [6, 6, 1080], [4, 3, 495], [2, 5, 260]], query = [])

    def test_no_value_passed_to_the_function(self):
        """test to make sure the function throws a TypeError when no values are passed
           to the knn_regression()."""
        with self.assertRaises(TypeError):
            knn_regression(n_neighbors = [], data = np.empty([4,3]), query = [])

    @classmethod
    def test_give_the_expected_result(cls):
        """One shot test using the known case of n nearest neighbours, data and the query. Should
           return 931.00"""
        assert np.isclose(knn_regression(5, [[8, 1, 530], [6, 2, 345], [5, 6, 1980], [4, 3, 900],
                                         [7, 9, 780], [0, -1, 900]], [3, -2]), 931.00)

    def test_n_neighbors_is_negative(self):
        """Edge test to make sure the function throws a ValueError when the n_neighbors
           is negative."""
        with self.assertRaises(ValueError):
            knn_regression(-1, [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495],
                           [2, 5, 260]], [5, 4])
    @classmethod
    def test_give_the_expected_result1(cls):
        """One shot test using the known case of n nearest neighbours, data and the query.
           Should return -40.00"""
        assert np.isclose(knn_regression(3, [[8, 1, 230], [6, 2, -745], [7, 6, 1980],
                          [4, 3, 395]], [6, 2]), -40.00)
    @classmethod
    def test_give_the_expected_result2(cls):
        """One shot test using the known case of n nearest neighbours, data and the query.
            Should return 773.3333"""
        assert np.isclose(knn_regression(3, [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495],
                           [2, 5, 260]], [5, 4]), 773.3333)

    def test_check_return_value_of_knn_regression_is_float_type(self):
        """test to make sure the returned value of the knn_regression function is float type."""
        self.assertIsInstance(knn_regression(3, [[3, 1, 230], [6, 2, 745], [6, 6, 1080],
                              [4, 3, 495], [2, 5, 260]], [5, 4]), float)

    def test_data_value_passed_to_the_function_is_not_list(self):
        """test to make sure the function throws a TypeError when data values passed to
           the knn_regression() is not list or array type."""
        with self.assertRaises(TypeError):
            knn_regression(3, "[[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495], [2, 5, 260]]",
                           [5, 4])
