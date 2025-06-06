import unittest
from unittest.mock import patch
from functions import augment_text_complex

class TestAugmentTextComplex(unittest.TestCase):
    @patch('functions.random.choice')
    @patch('functions.syn_aug.augment')
    def test_synonym_augmentation(self, mock_augment, mock_choice):
        mock_choice.return_value = 'synonym'
        mock_augment.return_value = "texto aumentado"
        result = augment_text_complex("texto original", 1)
        self.assertEqual(result, ["texto aumentado"])
    
    @patch('functions.random.choice')
    @patch('functions.swap_aug.augment')
    def test_swap_augmentation(self, mock_augment, mock_choice):
        mock_choice.return_value = 'swap'
        mock_augment.return_value = ["texto intercambiado"]
        result = augment_text_complex("texto original", 1)
        self.assertEqual(result, ["texto intercambiado"])
    
    @patch('functions.random.choice')
    def test_default_augmentation(self, mock_choice):
        mock_choice.return_value = 'invalid_type'
        result = augment_text_complex("texto original", 1)
        self.assertEqual(result, ["texto original"])
    
    @patch('functions.random.choice')
    @patch('functions.syn_aug.augment')
    def test_multiple_augments(self, mock_augment, mock_choice):
        mock_choice.return_value = 'synonym'
        mock_augment.return_value = "texto aumentado"
        result = augment_text_complex("texto original", 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, ["texto aumentado"] * 3)
    
    @patch('functions.random.choice')
    @patch('functions.syn_aug.augment')
    def test_empty_input(self, mock_augment, mock_choice):
        mock_choice.return_value = 'synonym'
        mock_augment.return_value = ""
        result = augment_text_complex("", 1)
        self.assertEqual(result, [""])
    
    @patch('functions.random.choice')
    @patch('functions.swap_aug.augment')
    def test_special_characters(self, mock_augment, mock_choice):
        mock_choice.return_value = 'swap'
        mock_augment.return_value = ["@user #hashtag"]
        result = augment_text_complex("@user #hashtag", 1)
        self.assertEqual(result, ["@user #hashtag"])

if __name__ == "__main__":
    unittest.main()