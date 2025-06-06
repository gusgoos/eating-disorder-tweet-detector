import unittest
from unittest.mock import patch, MagicMock 
from functions import preprocess_text_bow

class TestPreprocessTextBOW(unittest.TestCase):
    @patch('functions.convert_emojis')
    @patch('functions.split_hashtag')
    @patch('functions.nlp')
    def test_standard_processing(self, mock_nlp, mock_split, mock_convert):
        mock_convert.return_value = "test content"
        mock_split.return_value = "split tag"
        
        mock_doc = MagicMock()
        mock_token1 = MagicMock(lemma_="test", is_stop=False, is_punct=False, like_num=False)
        mock_token2 = MagicMock(lemma_="content", is_stop=False, is_punct=False, like_num=False)
        mock_doc.__iter__.return_value = [mock_token1, mock_token2]
        mock_nlp.return_value = mock_doc
        
        result = preprocess_text_bow("Original #test @user http://test.com 123")
        self.assertEqual(result, "test content")
    
    @patch('functions.convert_emojis')
    @patch('functions.split_hashtag')
    @patch('functions.nlp')
    def test_mixed_language_text(self, mock_nlp, mock_split, mock_convert):
        mock_convert.return_value = "hello mundo"
        mock_split.return_value = "hashtag"
        
        mock_doc = MagicMock()
        mock_token1 = MagicMock(lemma_="hello", is_stop=False, is_punct=False, like_num=False)
        mock_token2 = MagicMock(lemma_="mundo", is_stop=False, is_punct=False, like_num=False)
        mock_doc.__iter__.return_value = [mock_token1, mock_token2]
        mock_nlp.return_value = mock_doc
        
        result = preprocess_text_bow("Hello #mundo @user123")
        self.assertEqual(result, "hello mundo")
    
    @patch('functions.convert_emojis')
    @patch('functions.split_hashtag')
    @patch('functions.nlp')
    def test_preserved_punctuation(self, mock_nlp, mock_split, mock_convert):
        mock_convert.return_value = "test content"
        mock_split.return_value = "split tag"
        
        mock_doc = MagicMock()
        mock_token1 = MagicMock(lemma_="test", is_stop=False, is_punct=False, like_num=False)
        mock_token2 = MagicMock(lemma_="don't", is_stop=False, is_punct=False, like_num=False)
        mock_doc.__iter__.return_value = [mock_token1, mock_token2]
        mock_nlp.return_value = mock_doc
        
        result = preprocess_text_bow("Test don't")
        self.assertEqual(result, "test don't")

if __name__ == "__main__":
    unittest.main()