"""This file contains the pre-processing methods for the toxicity data."""

import logging
from pathlib import Path
import re

from deep_translator import GoogleTranslator
import emoji
import ftfy
import nltk
from nltk.corpus import stopwords
import pandas
from symspellpy.symspellpy import SymSpell, Verbosity

path_train = Path(__file__).parent / "data" / "train.csv"
path_train_corrected = Path(__file__).parent / "data" / "train_corrected.csv"

execute_translation = False


class TranslationService:
    """Instance of a translation service."""

    def __init__(self):
        pass

    def translate_text(row: pandas.Series) -> pandas.Series:
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
            source='es', target='en').translate(row['text_processed_es']
            )

        # translate to French
        french_translation = GoogleTranslator(
            source='es', target='fr').translate(row['text_processed_es']
            )

        return pandas.Series([english_translation, french_translation])

    except Exception as e:
        print(f"Error en la traducción: {e}")
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
    [df_train_no_wrong_translations, corrected_translations], ignore_index=True
    ).drop_duplicates(subset=['text_processed_es']).drop_duplicates(
        subset=['text_processed_en']
        ).drop_duplicates(
            subset=['text_processed_fr']
            ).dropna(
                subset=[
                    'text_processed_es',
                    'text_processed_en',
                    'text_processed_fr'
                    ]
                )
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
    df_train_corrected.duplicated(
        subset=['id', 'text_processed_en'], keep=False
        )
    ]
    wrong_french_translation = df_train_corrected[
        df_train_corrected.duplicated(
            subset=['id', 'text_processed_fr'], keep=False
            )
        ]
    null_data = df_train_corrected[
        df_train_corrected['text_processed_en'].isna() |
        df_train_corrected['text_processed_fr'].isna()
        ]

	assert (
		len(wrong_english_translation) == 
		len(wrong_french_translation) == 
		len(null_data) == 
		0
		)

def preprocess_text(text: str) -> str:
    """
    Apply certain transformations to a string to preprocess it.

    The given string gets emojis, special chars, links and multiple blanks
    removed. The hashtags are converted into individual words using their inner
    capitalization. If no capitalization is found, the library SymSpell is used
    to find the most likely inner words forming the hashtag.

    Args:
        text (str): the given text

    Returns:
        text_processed (str): the resulting string obtained after the
        transformation
    """
    if isinstance(text, str):
      # remove emoji
      text = emoji.replace_emoji(text, replace='')

      # replace "\n" with blank
      text = re.sub(r'\n', ' ', text)

      # remove links
      text = re.sub(r'http\S+|www\S+|pic.\S+', '', text)

      def replace_hashtags(match):
          # function to obtain inner words from a string
          words = re.findall(
            r"[A-Za-zÁÉÍÓÚÑÜ][a-záéíóúñü]*|[A-ZÁÉÍÓÚÑÜ]+(?![a-záéíóúñü])|\d+",
            match.group(1)
            )

          if len(words) == 1:
            # if hashtag is formed without capitalization, try to split
            hashtag_text = match.group(1)

            suggestions = sym_spell.word_segmentation(hashtag_text)

            words = [segment for segment in suggestions.corrected_string.split()]

          return ' '.join(words)

      # replace hashtags with its inner words
      text = re.sub(r'#(\w+)', replace_hashtags, text)

      # remove special caracters
      text = re.sub(
          r'[^a-záéíóúñüA-ZÁÉÍÓÚÑÜ\d\s\(\)\[\]\{\}\?\!\.,;:\'\"]',
          '',
          text
          )

      # replace multiple blank spaces
      text = re.sub(r'\s+', ' ', text)

      text = text.strip()

    return text


if __name__ == "__main__":
	df_train = load_data(path_train)

    # fix text in all languages keeping NaNs for now
    df_train['text'] = df_train['text'].apply(ftfy.fix_text)
    df_train['english'] = df_train['english'].apply(
        lambda x : ftfy.fix_text(x) if isinstance(x, str) else numpy.nan
        )
    df_train['french'] = df_train['french'].apply(
        lambda x : ftfy.fix_text(x) if isinstance(x, str) else numpy.nan
        )

    # create a Symspell instance with an Spanish dictionary
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary('es_dict.txt', term_index=0, count_index=1)

    # apply preprocessing to all texts of the different languages
    df_train['text_processed_es'] = df_train.apply(
        lambda x: preprocess_text(x['text']),
        axis=1
        )
    df_train['text_processed_en'] = df_train.apply(
        lambda x: preprocess_text(x['english']),
        axis=1
        )
    df_train['text_processed_fr'] = df_train.apply(
        lambda x: preprocess_text(x['french']),
        axis=1
        )

	df_train_corrected = correct_translations(df_train)

	if check_translation_are_fixed(df_train_corrected):
		# save training data after pre-processing
		df_train_corrected.to_csv(
			path_train_corrected,
			encoding="utf-8",
			index=False
			)
	else:
		logging.error('Something went wrong when fixing translations on data.')
