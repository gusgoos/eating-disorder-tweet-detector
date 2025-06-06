import unittest
import pandas as pd
import numpy as np
from functions import process_llm_broken_predictions

class TestProcessLLMPredictions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'label_enc': [0, 1, 0, 1, 0, 1],
            'prob_col': [0.2, 0.8, -1.0, 0.9, -1.0, 0.7]
        })

    def test_remove_invalid(self):
        y_true, y_prob = process_llm_broken_predictions(self.df, 'prob_col', -1)
        np.testing.assert_array_equal(y_true, [0, 1, 1, 1])
        np.testing.assert_array_equal(y_prob, [0.2, 0.8, 0.9,0.7])

    def test_replace_with_zero(self):
        y_true, y_prob = process_llm_broken_predictions(self.df, 'prob_col', 0)
        np.testing.assert_array_equal(y_true, [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(y_prob, [0.2, 0.8, 0.0, 0.9, 0.0, 0.7])

    def test_replace_with_one(self):
        y_true, y_prob = process_llm_broken_predictions(self.df, 'prob_col', 1)
        np.testing.assert_array_equal(y_true, [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(y_prob, [0.2, 0.8, 1.0, 0.9, 1.0, 0.7])

    def test_all_valid_predictions(self):
        df = pd.DataFrame({
            'label_enc': [0, 1, 0],
            'prob_col': [0.1, 0.9, 0.3]
        })
        y_true, y_prob = process_llm_broken_predictions(df, 'prob_col', -1)
        np.testing.assert_array_equal(y_true, [0, 1, 0])
        np.testing.assert_array_equal(y_prob, [0.1, 0.9, 0.3])

    def test_all_invalid_predictions(self):
        df = pd.DataFrame({
            'label_enc': [0, 1, 0],
            'prob_col': [-1.0, -1.0, -1.0]
        })
        y_true, y_prob = process_llm_broken_predictions(df, 'prob_col', -1)
        self.assertEqual(len(y_true), 0)
        self.assertEqual(len(y_prob), 0)
        
        y_true, y_prob = process_llm_broken_predictions(df, 'prob_col', 0)
        np.testing.assert_array_equal(y_prob, [0.0, 0.0, 0.0])

    def test_mixed_strategies(self):
        df = pd.DataFrame({
            'label_enc': [0, 1, 0, 1, 0],
            'prob_col': [0.4, -1.0, 0.2, -1.0, 0.5]
        })
        y_true, y_prob = process_llm_broken_predictions(df, 'prob_col', 1)
        np.testing.assert_array_equal(y_prob, [0.4, 1.0, 0.2, 1.0, 0.5])

if __name__ == "__main__":
    unittest.main()