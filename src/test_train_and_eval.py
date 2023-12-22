import unittest
from unittest.mock import MagicMock
import pandas
import torch
from torch.utils.data import DataLoader
from train_and_eval import (
    ToxicityDataset,
    read_data,
    create_data_loader,
    train_epoch,
    validate_epoch,
    evaluate_model,
)

class TestTrainAndEval(unittest.TestCase):

    def setUp(self):
        # Initialize any common variables needed for the tests
        self.sample_data = {
            'text_processed_es': ['texto ejemplo'],
            'text_processed_en': ['sample text'],
            'text_processed_fr': ['exemple de texte'],
            'label': [0],
        }
        self.df_train = pandas.DataFrame(self.sample_data)
        self.train_loader = create_data_loader(
          self.df_train,
          ['text_processed_es', 'text_processed_en', 'text_processed_fr']
          )

    def test_toxicity_dataset(self):
        """Test ToxicityDataset class."""
        dataset = ToxicityDataset(self.sample_data['text_processed_es'], self.sample_data['label'])
        self.assertEqual(len(dataset), len(self.sample_data['label']))
        sample = dataset[0]
        self.assertTrue(isinstance(sample['input_ids'], torch.Tensor))
        self.assertTrue(isinstance(sample['attention_mask'], torch.Tensor))
        self.assertTrue(isinstance(sample['label'], torch.Tensor))

    def test_read_data(self):
        """Test read_data function."""
        df_train, _, _, label = read_data(self.df_train, None)
        self.assertTrue('text_processed_es' in df_train.columns)
        self.assertTrue('text_processed_en' in df_train.columns)
        self.assertTrue('text_processed_fr' in df_train.columns)
        self.assertTrue('label' in df_train.columns)
        self.assertEqual(len(df_train), len(label))

    def test_create_data_loader(self):
        """Test create_data_loader function."""
        self.assertTrue(isinstance(self.train_loader, DataLoader))

    def test_train_epoch(self):
        # Test train_epoch function
        model = MagicMock()
        criterion = MagicMock()
        optimizer = MagicMock()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_examples = len(self.df_train)
        accuracy, loss = train_epoch(
          model,
          criterion,
          self.train_loader,
          optimizer,
          device,
          n_examples
          )
        self.assertTrue(isinstance(accuracy, torch.Tensor))
        self.assertTrue(isinstance(loss, float))

    def test_validate_epoch(self):
        """Test validate_epoch function."""
        model = MagicMock()
        criterion = MagicMock()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_examples = len(self.df_train)
        accuracy, loss = validate_epoch(
          model,
          criterion,
          self.train_loader,
          device,
          n_examples
          )
        self.assertTrue(isinstance(accuracy, torch.Tensor))
        self.assertTrue(isinstance(loss, float))

    def test_evaluate_model(self):
        """Test evaluate_model function."""
        model = MagicMock()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y_true, y_pred = evaluate_model(model, self.train_loader, device)
        self.assertTrue(isinstance(y_true, list))
        self.assertTrue(isinstance(y_pred, list))
        self.assertEqual(len(y_true), len(y_pred))

if __name__ == '__main__':
    unittest.main()
