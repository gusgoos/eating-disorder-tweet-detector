import unittest
from unittest.mock import patch
from functions import convert_emojis

class TestConvertEmojis(unittest.TestCase):
    @patch('functions.emoji.demojize')
    def test_single_emoji(self, mock_demojize):
        mock_demojize.return_value = "Hello :smiling_face:"
        result = convert_emojis("Hello 😊")
        self.assertEqual(result, "Hello :smiling_face:")
        mock_demojize.assert_called_once_with("Hello 😊")

    @patch('functions.emoji.demojize')
    def test_multiple_emojis(self, mock_demojize):
        mock_demojize.return_value = "I :red_heart: Python :thumbs_up:"
        result = convert_emojis("I ❤️ Python 👍")
        self.assertEqual(result, "I :red_heart: Python :thumbs_up:")

    @patch('functions.emoji.demojize')
    def test_emoji_only(self, mock_demojize):
        mock_demojize.return_value = ":fire::rocket::star:"
        result = convert_emojis("🔥🚀⭐")
        self.assertEqual(result, ":fire::rocket::star:")

    @patch('functions.emoji.demojize')
    def test_mixed_content(self, mock_demojize):
        mock_demojize.return_value = "Alert! :warning: Important :exclamation_mark:"
        result = convert_emojis("Alert! ⚠️ Important ❗")
        self.assertEqual(result, "Alert! :warning: Important :exclamation_mark:")

    @patch('functions.emoji.demojize')
    def test_emoji_with_skin_tones(self, mock_demojize):
        mock_demojize.return_value = ":thumbs_up_light_skin_tone: :thumbs_up_dark_skin_tone:"
        result = convert_emojis("👍🏻 👍🏿")
        self.assertEqual(result, ":thumbs_up_light_skin_tone: :thumbs_up_dark_skin_tone:")

    @patch('functions.emoji.demojize')
    def test_empty_string(self, mock_demojize):
        mock_demojize.return_value = ""
        result = convert_emojis("")
        self.assertEqual(result, "")
        mock_demojize.assert_called_once_with("")

    def test_none_input(self):
        with self.assertRaises(TypeError):
            convert_emojis(None)

    def test_integer_input(self):
        with self.assertRaises(TypeError):
            convert_emojis(123)

if __name__ == "__main__":
    unittest.main()