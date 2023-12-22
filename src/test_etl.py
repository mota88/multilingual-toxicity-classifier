"""This file contains the unitary tests for etl.py."""

import pandas
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

from etl import (
    TranslationService,
    load_data,
    correct_translations,
    check_translation_are_fixed,
    preprocess_text
    )

class TestEtl(unittest.TestCase):

    def setUp(self):
        # Set up any necessary objects or configurations
        self.data_dir = Path(__file__).parent / "data"
        self.train_data_path = self.data_dir / "train.csv"

    def test_load_data(self):
        """Test function that loads data."""
        df_train = load_data(self.train_data_path)
        
        self.assertEqual(len(df_train), 11000)

    def test_correct_translations(self):
        """Test function that correct translations."""
        # mock TranslationService to simulate translations
        translation_service_mock = Mock()
        translation_service_mock.translate_text.return_value = pandas.Series(
            ["English", "French"]
            )

        # mock the execute_translation flag to True
        with patch("etl.execute_translation", True):
            df_train = pandas.DataFrame({
                'id': [1, 1, 1],
                'text': ['Hola', 'Hola', 'Bonjour'],
                'english': [None, None, None],
                'french': [None, None, None]
            })

            df_corrected = correct_translations(
                df_train,
                translation_service_mock
                )

            # ensure translations are applied
            translation_service_mock.translate_text.assert_called()

            # ensure corrected dataframe has no duplicates
            self.assertEqual(
                len(df_corrected.duplicated(
                    subset=['id', 'english'], keep=False)
                ),
                len(df_corrected.duplicated(
                    subset=['id', 'english'], keep=False))
                )

    def test_preprocess_text(self):
        """Test the preprocess_text function."""
        text = 'This is a sample text with stopwords and a link https://edu.com'
        processed_text = preprocess_text(text, 'english', min_length=3)
        expected_output = 'sample text stopwords link'
        self.assertEqual(processed_text, expected_output)


if __name__ == '__main__':
    unittest.main()
