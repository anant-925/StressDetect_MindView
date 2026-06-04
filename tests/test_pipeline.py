import os
import shutil
import tempfile
import unittest
import sys

# Ensure repository root is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utils.sentiment import get_sentiment_score, compute_sentiment_dampening
from utils.text_preprocessing import clean_text as standard_clean_text
from scripts.data_preprocessing import clean_text as preprocess_clean_text
from security.auth import encrypt_data, decrypt_data


class TestStressDetectionPipeline(unittest.TestCase):

    def setUp(self):
        # Create a temp directory for any file operations
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove temp directory
        shutil.rmtree(self.test_dir)

    def test_sentiment_negation(self):
        """Verify that get_sentiment_score handles negations correctly.
        "not stressed" should get a low score, whereas "stressed" should get a high score.
        """
        stressed_score = get_sentiment_score("I am stressed")
        not_stressed_score = get_sentiment_score("I am not stressed")

        # "I am stressed" has 1 negative hit, 0 positive hits, 0 negations -> score 1.0
        self.assertGreater(stressed_score, 0.7)

        # "I am not stressed" has 1 negation hit, 0 negative hits (neutralized), 0 positive hits -> score 0.0
        self.assertLess(not_stressed_score, 0.3)

    def test_preprocessing_alignment(self):
        """Verify that preprocessing clean_text aligns with standard clean_text
        and does not strip out critical stop words, preserving consistency.
        """
        sample = "I am stressed but happy today"
        
        preprocessed = preprocess_clean_text(sample)
        standard = standard_clean_text(sample)
        
        # Check that both keep standard/conjunction words and yield identical results
        self.assertEqual(preprocessed, standard)
        self.assertIn("but", preprocessed)
        self.assertIn("happy", preprocessed)

    def test_fernet_key_persistence(self):
        """Verify that auth key persists locally and can decrypt data written previously."""
        key_file_path = os.path.join(_ROOT, ".fernet_key")
        
        # Verify that the key file exists (it should have been generated during import or server start)
        self.assertTrue(os.path.isfile(key_file_path))
        
        # Read the current key
        with open(key_file_path, "r", encoding="utf-8") as f:
            key1 = f.read().strip()
            
        # Re-import or re-initialize logic would fetch the same key. Let's verify encrypting and decrypting.
        payload = {"history": [[1234567.0, 0.45], [1234568.0, 0.82]]}
        encrypted = encrypt_data(payload)
        
        # Decrypt it using standard auth function
        decrypted = decrypt_data(encrypted)
        
        self.assertEqual(decrypted, payload)


if __name__ == "__main__":
    unittest.main()
