import unittest
from unittest.mock import patch
from functions import split_hashtag

class TestSplitHashtag(unittest.TestCase):
    def test_camel_case_split(self):
        result = split_hashtag("#HelloWorld")
        self.assertEqual(result, "hello world")
        
    def test_multi_word_camelcase(self):
        result = split_hashtag("#ThisIsATest")
        self.assertEqual(result, "this is atest")
        
    @patch('functions.segment')
    def test_wordsegment_split(self, mock_segment):
        mock_segment.return_value = ["artificial", "intelligence"]
        result = split_hashtag("#ArtificialIntelligence")
        self.assertEqual(result, "artificial intelligence")
        
    def test_mixed_camelcase_with_numbers(self):
        result = split_hashtag("#Python3IsAwesome")
        self.assertEqual(result, "python3is awesome")
        
    def test_single_letter_handling(self):
        result = split_hashtag("#AITechnology")
        self.assertEqual(result, "a i technology")
        
    def test_all_uppercase_handling(self):
        result = split_hashtag("#ALLUPPERCASE")
        self.assertEqual(result, "all uppercase")
        
    def test_consecutive_uppercase(self):
        result = split_hashtag("#HTTPRequest")
        self.assertEqual(result, "httprequest")
        
    @patch('functions.segment')
    def test_segmentation_error(self, mock_segment):
        mock_segment.side_effect = Exception("Error")
        result = split_hashtag("#error")
        self.assertEqual(result, "error")
        
    @patch('functions.segment')
    def test_empty_string_after_stripping(self, mock_segment):
        mock_segment.side_effect = Exception("Empty input")
        result = split_hashtag("###")
        self.assertEqual(result, "")

if __name__ == "__main__":
    unittest.main()