import unittest
from functions import calculate_uppercase_ratio

class TestUpperCaseRatio(unittest.TestCase):
    def test_all_uppercase(self):
        result = calculate_uppercase_ratio("HELLO WORLD")
        self.assertEqual(result, 1.0)
    
    def test_mixed_case(self):
        result = calculate_uppercase_ratio("Hello WORLD")
        self.assertEqual(result, 0.5)
    
    def test_no_uppercase(self):
        result = calculate_uppercase_ratio("hello world")
        self.assertEqual(result, 0.0)
    
    def test_empty_string(self):
        result = calculate_uppercase_ratio("")
        self.assertEqual(result, 0.0)
    
    # New test cases below
    def test_single_uppercase_word(self):
        result = calculate_uppercase_ratio("Normal TEXT here and")
        self.assertEqual(result, 0.25) 
    
    def test_numbers_and_symbols(self):
        result = calculate_uppercase_ratio("TEST @#$")
        self.assertEqual(result, 0.5)  
    
    def test_unicode_uppercase(self):
        result = calculate_uppercase_ratio("CAFÉ ÉPICÉ")
        self.assertEqual(result, 1.0)  
    
    def test_mixed_unicode_case(self):
        result = calculate_uppercase_ratio("México CITY")
        self.assertEqual(result, 0.5)  

if __name__ == "__main__":
    unittest.main()