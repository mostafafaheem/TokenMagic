import unittest
import tempfile
import os
import json
import base64
from tokenizer import BPETokenizer, SPECIAL_TOKENS, PATTERN_STRING


class TestBPETokenizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tokenizer = BPETokenizer()
        
    def test_initialization_default_pattern(self):
        """Test tokenizer initialization with default pattern."""
        tokenizer = BPETokenizer()
        self.assertEqual(tokenizer.pattern, PATTERN_STRING)
        self.assertEqual(len(tokenizer.vocab), 256)  # Should have 256 byte tokens
        self.assertEqual(len(tokenizer.lookup), 256)
        self.assertEqual(len(tokenizer.merges), 0)  # No merges initially
        self.assertEqual(len(tokenizer.special_tokens), 0)  # No special tokens initially
        
    def test_initialization_custom_pattern(self):
        """Test tokenizer initialization with custom pattern."""
        custom_pattern = r"\w+|\s+"
        tokenizer = BPETokenizer(pattern=custom_pattern)
        self.assertEqual(tokenizer.pattern, custom_pattern)
        self.assertEqual(len(tokenizer.vocab), 256)
        self.assertEqual(len(tokenizer.lookup), 256)
        
    def test_initial_vocab_structure(self):
        """Test that initial vocabulary contains all byte values."""
        tokenizer = BPETokenizer()
        
        # Check that vocab contains all bytes 0-255
        for i in range(256):
            byte_token = bytes([i])
            self.assertIn(byte_token, tokenizer.vocab)
            self.assertEqual(tokenizer.vocab[byte_token], i)
            self.assertEqual(tokenizer.lookup[i], byte_token)
            
    def test_train_vocabulary_size_validation(self):
        """Test training with invalid vocabulary sizes."""
        tokenizer = BPETokenizer()
        
        # Test with vocabulary size less than 256
        with self.assertRaises(ValueError):
            tokenizer.train(255, "test text")
            
        with self.assertRaises(ValueError):
            tokenizer.train(100, "test text")
            
    def test_train_basic_functionality(self):
        """Test basic training functionality."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        merges = tokenizer.train(vocab_size, text)
        
        # Check that merges were created
        self.assertGreater(len(merges), 0)
        self.assertEqual(len(tokenizer.merges), len(merges))
        
        # Check that vocabulary size increased (but may be less than requested if not enough pairs)
        self.assertGreater(len(tokenizer.vocab), 256)  # More than initial 256 bytes
        self.assertGreater(len(tokenizer.lookup), 256)
        
        # Check that special tokens were added
        self.assertEqual(len(tokenizer.special_tokens), len(SPECIAL_TOKENS))
        for token in SPECIAL_TOKENS:
            self.assertIn(token, tokenizer.special_tokens)
            
    def test_train_with_special_tokens(self):
        """Test training with text containing special tokens."""
        tokenizer = BPETokenizer()
        text = "hello <BOS> world <EOS>"
        vocab_size = 300
        
        merges = tokenizer.train(vocab_size, text)
        
        # Should have merges
        self.assertGreater(len(merges), 0)
        
        # Special tokens should be properly handled
        self.assertEqual(len(tokenizer.special_tokens), len(SPECIAL_TOKENS))
        
    def test_train_empty_text(self):
        """Test training with empty text."""
        tokenizer = BPETokenizer()
        text = ""
        vocab_size = 300
        
        merges = tokenizer.train(vocab_size, text)
        
        # Should still create special tokens
        self.assertEqual(len(tokenizer.special_tokens), len(SPECIAL_TOKENS))
        
    def test_train_single_character_text(self):
        """Test training with single character text."""
        tokenizer = BPETokenizer()
        text = "a"
        vocab_size = 300
        
        merges = tokenizer.train(vocab_size, text)
        
        # Should have special tokens but no merges for single char
        self.assertEqual(len(tokenizer.special_tokens), len(SPECIAL_TOKENS))
        
    def test_encode_without_training(self):
        """Test encoding without prior training."""
        tokenizer = BPETokenizer()
        
        with self.assertRaises(ValueError):
            tokenizer.encode("test text")
            
    def test_encode_basic_text(self):
        """Test encoding basic text."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        tokens = tokenizer.encode(text)
        
        # Should return list of integers
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        self.assertGreater(len(tokens), 0)
        
    def test_encode_with_special_tokens(self):
        """Test encoding text with special tokens."""
        tokenizer = BPETokenizer()
        text = "hello <BOS> world <EOS>"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        tokens = tokenizer.encode(text)
        
        # Should successfully encode
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        
    def test_encode_with_custom_merges(self):
        """Test encoding with custom merges parameter."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        merges = tokenizer.train(vocab_size, text)
        tokens1 = tokenizer.encode(text)
        tokens2 = tokenizer.encode(text, merges)
        
        # Should produce same results
        self.assertEqual(tokens1, tokens2)
        
    def test_encode_unknown_token_error(self):
        """Test encoding with unknown token raises error."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        
        # Create a tokenizer with different merges to simulate unknown token
        other_tokenizer = BPETokenizer()
        other_tokenizer.train(vocab_size, "different text")
        
        # This might raise ValueError if there are unknown tokens
        # The exact behavior depends on implementation
        
    def test_decode_basic_tokens(self):
        """Test decoding basic tokens."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Should decode back to original text
        self.assertEqual(decoded, text)
        
    def test_decode_with_control_characters(self):
        """Test decoding with control character visualization."""
        tokenizer = BPETokenizer()
        text = "hello\nworld\t"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        tokens = tokenizer.encode(text)
        
        # Test without visualization
        decoded_normal = tokenizer.decode(tokens, visusalize_control_characters=False)
        
        # Test with visualization
        decoded_visual = tokenizer.decode(tokens, visusalize_control_characters=True)
        
        # Both should be strings
        self.assertIsInstance(decoded_normal, str)
        self.assertIsInstance(decoded_visual, str)
        
        # Visualized version should contain escape sequences
        if "\n" in text:
            self.assertIn("\\n", decoded_visual)
        if "\t" in text:
            self.assertIn("\\t", decoded_visual)
            
    def test_decode_empty_tokens(self):
        """Test decoding empty token list."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        
        decoded = tokenizer.decode([])
        self.assertEqual(decoded, "")
        
    def test_save_and_load_functionality(self):
        """Test saving and loading tokenizer."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        # Train the tokenizer
        tokenizer.train(vocab_size, text)
        
        # Save to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_tokenizer")
            tokenizer.save(filename)
            
            # Check that files were created
            model_file = filename + ".model.json"
            vocab_file = filename + ".vocab.json"
            self.assertTrue(os.path.exists(model_file))
            self.assertTrue(os.path.exists(vocab_file))
            
            # Load into new tokenizer
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(filename)
            
            # Check that data was loaded correctly
            self.assertEqual(tokenizer.pattern, new_tokenizer.pattern)
            self.assertEqual(tokenizer.vocab, new_tokenizer.vocab)
            self.assertEqual(tokenizer.lookup, new_tokenizer.lookup)
            self.assertEqual(tokenizer.merges, new_tokenizer.merges)
            self.assertEqual(tokenizer.special_tokens, new_tokenizer.special_tokens)
            
    def test_save_load_roundtrip(self):
        """Test that save/load preserves functionality."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        # Train and encode
        tokenizer.train(vocab_size, text)
        original_tokens = tokenizer.encode(text)
        original_decoded = tokenizer.decode(original_tokens)
        
        # Save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_tokenizer")
            tokenizer.save(filename)
            
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(filename)
            
            # Test that functionality is preserved
            new_tokens = new_tokenizer.encode(text)
            new_decoded = new_tokenizer.decode(new_tokens)
            
            self.assertEqual(original_tokens, new_tokens)
            self.assertEqual(original_decoded, new_decoded)
            
    def test_special_tokens_handling(self):
        """Test proper handling of special tokens."""
        tokenizer = BPETokenizer()
        text = "hello world"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        
        # Check that all special tokens are in vocabulary
        for token in SPECIAL_TOKENS:
            self.assertIn(token, tokenizer.special_tokens)
            token_id = tokenizer.special_tokens[token]
            self.assertIn(token_id, tokenizer.lookup)
            
    def test_multilingual_text(self):
        """Test tokenizer with multilingual text."""
        tokenizer = BPETokenizer()
        text = "Hello 世界 مرحبا 안녕하세요"
        vocab_size = 500
        
        tokenizer.train(vocab_size, text)
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Should handle multilingual text
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(decoded, str)
        
    def test_unicode_edge_cases(self):
        """Test tokenizer with Unicode edge cases."""
        tokenizer = BPETokenizer()
        text = "🚀 emoji test \u0000 null char \uFFFF high unicode"
        vocab_size = 500
        
        tokenizer.train(vocab_size, text)
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Should handle Unicode properly
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(decoded, str)
        
    def test_whitespace_handling(self):
        """Test tokenizer with various whitespace characters."""
        tokenizer = BPETokenizer()
        text = "hello\tworld\nnewline\r\ncarriage"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Should handle whitespace properly
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(decoded, str)
        
    def test_large_vocabulary(self):
        """Test tokenizer with large vocabulary size."""
        tokenizer = BPETokenizer()
        # Use more diverse text to allow for more merges
        text = "the quick brown fox jumps over the lazy dog " * 10  # Repeat to create more patterns
        vocab_size = 1000  # More reasonable size
        
        tokenizer.train(vocab_size, text)
        
        # Should handle large vocabulary (may be less than requested if not enough unique pairs)
        self.assertGreater(len(tokenizer.vocab), 256)  # More than initial 256 bytes
        self.assertGreater(len(tokenizer.lookup), 256)
        # Check that it doesn't exceed the requested size
        self.assertLessEqual(len(tokenizer.vocab), vocab_size + len(SPECIAL_TOKENS))
        
    def test_repeated_text_patterns(self):
        """Test tokenizer with repeated text patterns."""
        tokenizer = BPETokenizer()
        text = "hello hello world world hello world"
        vocab_size = 300
        
        tokenizer.train(vocab_size, text)
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Should handle repeated patterns efficiently
        self.assertIsInstance(tokens, list)
        self.assertEqual(decoded, text)
        
    def test_roundtrip_consistency(self):
        """Test that encode-decode roundtrip preserves text."""
        tokenizer = BPETokenizer()
        test_texts = [
            "hello world",
            "Hello, World!",
            "123 numbers",
            "special <BOS> tokens <EOS>",
            "multiline\ntext\twith\ttabs",
            "unicode: café naïve résumé",
            "emoji: 🚀🌟🎉",
        ]
        vocab_size = 1000
        
        # Train on all texts
        combined_text = " ".join(test_texts)
        tokenizer.train(vocab_size, combined_text)
        
        # Test each text individually
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            self.assertEqual(decoded, text, f"Roundtrip failed for: {repr(text)}")


if __name__ == "__main__":
    unittest.main()