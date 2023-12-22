"""This file contains the pre-processing methods for the toxicity data."""

import logging
from pathlib import Path
import re

from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
import pandas

path_train = Path(__file__).parent / "data" / "train.csv"
path_train_corrected = Path(__file__).parent / "data" / "train_corrected.csv"

execute_translation = False


class TranslationService:
    """Instance of a translation service."""

    def __init__(self):
        pass

    def translate_text(self, row: pandas.Series) -> pandas.Series:
        """
        Translate the given text from Spanish to English and French.

        The function gets a pandas row as input and returns a pandas series with the
        translations.

        Args:
            row (pandas.Series): the row with the Spanish text to be translated

        Returns:
            pandas.Series: a pandas series with the translations or None if an
            exception occurs
        """
        
        try:
            # translate to English
            english_translation = GoogleTranslator(
                source='es', target='en').translate(row['text']
                )

            # translate to French
            french_translation = GoogleTranslator(
                source='es', target='fr').translate(row['text']
                )
            
            return pandas.Series([english_translation, french_translation])
        
        except Exception as e:
            logging.error(f"Error en la traducción: {e}")
            return pandas.Series([None, None])


def load_data(train_data_path: Path) -> pandas.DataFrame:
	"""
	Load train data.

	Args:
		train_data_path (Path): path to train data file
		test_data_path (Path): path to test data file

	Return:
		train_dataframe (pandas.DataFrame): dataframe containing the data
	"""
	train_dataframe = pandas.read_csv(train_data_path)

	return train_dataframe

def correct_translations(training_data: pandas.DataFrame) -> pandas.DataFrame:
    """
    Correct wrong translations and null values.

    The function replaces all wrong (duplicated rows) translations and null 
    values in the English and French columns of the data with translations 
    obtained from the function 'translate_text' or loaded from a data file.

    Args:
        training_data (pandas.DataFrame): data with wrong translations or nulls

    Returns:
        df_train_corrected (pandas.DataFrame): dataframe with corrected 
        translations
    """
    # get rows with wrong translation
    wrong_english_translation = training_data[
        training_data.duplicated(subset=['id', 'english'], keep=False)
    ]
    wrong_french_translation = training_data[
        training_data.duplicated(subset=['id', 'french'], keep=False)
    ]

    # get rows with null values in the translation fields
    null_data = training_data[
        training_data['english'].isna() | training_data['french'].isna()
    ]

    # combine all rows with wrong or missing translations
    wrong_translations = pandas.concat(
        [wrong_english_translation, wrong_french_translation, null_data]
    ).drop_duplicates()

    # copy the wrong_translations dataframe to store the corrected translations
    corrected_translations = wrong_translations.copy()

    if execute_translation:
        translation_service = TranslationService()
        # apply translate_text function to the wrong_translations dataframe
        corrected_translations[['english', 'french']] = (
            wrong_translations.apply(translation_service.translate_text, axis=1)
        )
    else:
        corrected_translations = pandas.read_csv(
            Path(__file__).parent / "data" / "corrected_translations.csv"
        )

    df_train_no_wrong_translations = df_train[
        ~df_train.isin(wrong_translations)
    ].dropna()

    df_train_corrected = pandas.concat(
        [df_train_no_wrong_translations, corrected_translations],
        ignore_index=True
    ).drop_duplicates(subset=['id', 'text'])
    df_train_corrected['label'] = df_train_corrected['label'].astype(int)

    return df_train_corrected


def check_translation_are_fixed(training_data_corrected: pandas.DataFrame) -> bool:
	"""
	Check that translations have been correctly fixed.

	Args:
        training_data_corrected (pandas.DataFrame): data with corrected 
        translations

    Returns:
        bool: bool indicating whether the problem has been fixed or not
	"""
	wrong_english_translation = df_train_corrected[
	    df_train_corrected.duplicated(subset=['id', 'english'], keep=False)
	    ]
	wrong_french_translation = df_train_corrected[
	    df_train_corrected.duplicated(subset=['id', 'french'], keep=False)
	    ]
	null_data = df_train_corrected[
	    df_train_corrected['english'].isna() | 
	    df_train_corrected['french'].isna()
	    ]

	assert (
		len(wrong_english_translation) == 
		len(wrong_french_translation) == 
		len(null_data) == 
		0
		)

def preprocess_text(text: str, language: str, min_length: int = 2) -> str:
    """
    Apply certain transformations to a string to preprocess it.

    The given string is lowercased, links and stop-words are removed if found
    and shorter words than a given threshold are also removed. Different
    languages are expected, and the function needs this information to remove
    the relevant stop-words.

    Args:
        text (str): the given text
        language (str): string with the language of the text
        min_length (int): minimum length for a word not to be removed

    Returns:
        text_processed (str): the resulting string obtained after the 
        transformation
    """
    # text in lowercase
    text_lower = text.lower()

    # remove links
    text_no_links = re.sub(r'http\S+', '', text_lower)

    # tokenize text
    words = re.findall(r'\b\w+\b', text_no_links)

    # remove stop-words and short words
    stop_words = set(stopwords.words(language))
    words_filtered = [
        word for word in words
        if word not in stop_words and len(word) >= min_length
        ]

    # join filtered words to obtain processed text
    text_processed = ' '.join(words_filtered)

    return text_processed


if __name__ == "__main__":
	df_train = load_data(path_train)

	df_train_corrected = correct_translations(df_train)

	if check_translation_are_fixed(df_train_corrected):
		# apply preprocessing to all texts of the different languages
		df_train_corrected['text_processed_es'] = df_train_corrected.apply(
		    lambda x: preprocess_text(x['text'], 'spanish'),
		    axis=1
		    )
		df_train_corrected['text_processed_en'] = df_train_corrected.apply(
		    lambda x: preprocess_text(x['english'], 'english'),
		    axis=1
		    )
		df_train_corrected['text_processed_fr'] = df_train_corrected.apply(
		    lambda x: preprocess_text(x['french'], 'french'),
		    axis=1
		    )

		# save training data after pre-processing
		df_train_corrected.to_csv(
			path_train_corrected,
			encoding="utf-8",
			index=False
			)
	else:
		logging.error('Something went wrong when fixing translations on data.')
