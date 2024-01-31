"""This file contains training and eval methods of a toxicity classifier."""

from itertools import cycle
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot
import numpy
import pandas
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve, auc
    )
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_path = Path(__file__).parent / "data" / "train_corrected.csv"
test_data_path = Path(__file__).parent / "data" / "test.csv"
num_epochs = 4
model_path = Path(__file__).parent / "model" / "toxicity_model.bin"


class ToxicityDataset(Dataset):
    """Instance of a toxicity dataset."""

    def __init__(self, texts, labels, max_length=162):
        """Initialise the toxicity dataset."""
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        """Return length of dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return data into model form."""
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx])

        # tokenize texts
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }


def read_data(train_path: Path, test_path: Path
    ) -> tuple[(
    pandas.DataFrame,
    pandas.DataFrame,
    pandas.DataFrame,
    pandas.Series
    )]:
    """
    Read data and split into training and validation.

    Args:
        train_path (Path): path to train data file
        test_path (Path): path to test data file

    Return:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, 
        pandas.Series]: dataframes containing the data along with label column 
        from whole data
    """
    train_df = pandas.read_csv(train_path)
    test_df = pandas.read_csv(test_path)
    
    # divide data into train and validation
    train, validation = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42
        )
    train.reset_index(inplace=True)
    validation.reset_index(inplace=True)

    return df_train, df_validation, test_df, train_df['label']


def create_data_loader(
    df: pandas.DataFrame,
    columns: list,
    alternate_languages: bool=True,
    shuffle: bool=True
    ) -> DataLoader:
    """
    Uses the data to create a DataLoader for model consumption.

    Args:
      df (pandas.DataFrame): dataframe containing text and label data
      columns (list): list of dataframe columns names containing the text
      alternate_languages (bool): bool indicating whether or not to alternate
      the different languages
      shuffle (bool): bool indicating whether or not to shuffle the DataLoader

    Return:
      DataLoader: data loader for model consumption
    """
    list_of_texts = []
    list_of_labels = df['label'].tolist()

    if alternate_languages:
      # used for training and validation, not for evaluation
      num_rows = len(df)

      # create a cyclic iterator for the columns
      column_iterator = cycle(columns)

      # get only the text from one lang for each row, and alternate languages
      for i in range(num_rows):
        column = next(column_iterator)
        text_value = str(df.loc[i, column])
        list_of_texts.append(text_value)
    else:
      # concat Spanish, English and French texts
      list_of_texts = (
          df[columns[0]].tolist() +
          df[columns[1]].tolist() +
          df[columns[2]].tolist()
      )

      # repeat the labels 3 times (same labels for each language)
      list_of_labels = df['label'].tolist() * 3

    # create dataset
    dataset = ToxicityDataset(list_of_texts, list_of_labels)

    return DataLoader(dataset, batch_size=32, shuffle=shuffle)


def train_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.CrossEntropyLoss,
    dataloader: DataLoader,
    optimizer: torch.optim,
    device: str,
    n_examples: int
) -> tuple[torch.Tensor, numpy.float64]:
    """
    Runs one epoch of training for the model.

    Args:
      model (torch.nn.Module): model to be trained
      criterion (torch.nn.CrossEntropyLoss): loss function for class
      weights ponderation
      dataloader (DataLoader): dataloader for model consumption
      optimizer (torch.optim): model optimizer
      device (str): string indicating what device to be used
      n_examples (int): length of training dataset

    Return:
      tuple[torch.Tensor, numpy.float64]: train accuracy and loss of epoch
    """
    model.train()

    losses = []
    correct_predictions = 0

    # iterate over batches of the dataloader
    for batch in tqdm(dataloader, desc=f'Training'):
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['label'].to(device)
        }
        # compute the outputs of the model using the input data
        outputs = model(**inputs)

        # compute the loss comparing the outputs of the model and the labels
        # using the CrossEntropyLoss loss function
        loss = criterion(outputs.logits, inputs['labels'])
        losses.append(loss.cpu().item())

        # track the number of correct predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(
            predictions == batch['label'].to(device)
            )

        # back-propagate, optimize model parameters and re-start gradients
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return (
        correct_predictions.cpu().double() / n_examples,
        sum(losses) / len(losses)
        )


def validate_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.CrossEntropyLoss,
    dataloader: DataLoader,
    device: str,
    n_examples: int
) -> tuple[torch.Tensor, numpy.float64]:
    """
    Runs one epoch of validation for the model.

    Args:
      model (torch.nn.Module): model for validation
      criterion (torch.nn.CrossEntropyLoss): loss function for class
      weights ponderation
      dataloader (DataLoader): dataloader for model consumption
      device (str): string indicating what device to be used
      n_examples (int): length of validation dataset

    Return:
      tuple[torch.Tensor, numpy.float64]: validation accuracy and loss of epoch
    """
    model.eval()

    losses = []
    correct_predictions = 0

    # avoid gradients computation during validation (not to update parameters)
    with torch.no_grad():
        # iterate over batches of the dataloader
        for batch in tqdm(dataloader, desc=f'Validación'):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['label'].to(device)
            }
            # compute the outputs of the model using the input data
            outputs = model(**inputs)

            # compute the loss comparing the outputs of the model and the labels
            # using the CrossEntropyLoss loss function
            loss = criterion(outputs.logits, inputs['labels'])
            losses.append(loss.cpu().item())

            # track the number of correct predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(
                predictions == batch['label'].to(device)
                )

    return (
        correct_predictions.cpu().double() / n_examples,
        sum(losses) / len(losses)
        )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str
) -> tuple[list, list]:
    """
    Evaluate the model on given data and return predictions and true labels.

    Args:
      model (torch.nn.Module): model for validation
      dataloader (DataLoader): dataloader for model consumption
      device (str): string indicating what device to be used

    Return:
      tuple[list, list]: list containing true labels and predictions
    """
    model.eval()
    y_true = []
    y_pred = []

    # avoid gradients computation during evaluation
    with torch.no_grad():
        # iterate over batches of the dataloader
        for batch in tqdm(dataloader, desc=f'Evaluación'):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['label'].to(device)
            }
            # compute the outputs of the model using the input data
            outputs = model(**inputs)

            # store the predictions and the true labels
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            y_true.extend(batch['label'].tolist())
            y_pred.extend(predictions.tolist())

    return y_true, y_pred


if __name__ == "__main__":
    train, validation, test, label = read_data(train_data_path, test_data_path)

    # compute weights classes to balance during training
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=numpy.unique(df_train_corrected['label']),
        y=numpy.array(df_train_corrected['label'])
        )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    # create dataloaders for all data
    train_dataloader = create_data_loader(
        train,
        ['text_processed_es', 'text_processed_en', 'text_processed_fr']
        )
    validation_dataloader = create_data_loader(
        validation,
        ['text_processed_es', 'text_processed_en', 'text_processed_fr'],
        shuffle=False
        )
    test_dataloader = create_data_loader(
        df_test,
        ['text', 'english', 'french'],
        alternate_languages=False,
        shuffle=False
        )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',
        num_labels=2
        ).to(device)
    model.classifier = torch.nn.Sequential(
    torch.nn.Linear(768, 384),
    torch.nn.Linear(384, 2),
    torch.nn.ReLU()
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        reduction='mean'
        )

    # training loop
    train_accuracies, train_losses = [], []
    val_accuracies, val_losses = [], []
    best_accuracy = 0

    for epoch in range(num_epochs):
      print(f"Epoch {epoch + 1}/{num_epochs}")
      print("-" * 10)

      train_acc, train_loss = train_epoch(
          model,
          criterion,
          train_dataloader,
          optimizer,
          device,
          len(train)
          )
      val_acc, val_loss = validate_epoch(
          model,
          criterion,
          validation_dataloader,
          device,
          len(validation)
          )

      train_accuracies.append(train_acc)
      train_losses.append(train_loss)
      val_accuracies.append(val_acc)
      val_losses.append(val_loss)

      print(
          f'Training Accuracy: {train_acc:.4f}, Training Loss: {train_loss:.4f}'
          )
      print(
          f'Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}'
          )

      if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_path)
            best_accuracy = val_acc

    # plot training results
    epochs = list(range(1, num_epochs + 1))

    matplotlib.pyplot.figure(figsize=(12, 5))

    matplotlib.pyplot.subplot(1, 2, 1)
    matplotlib.pyplot.plot(epochs, train_accuracies, label='Training Accuracy')
    matplotlib.pyplot.plot(epochs, val_accuracies, label='Validation Accuracy')
    matplotlib.pyplot.title('Training y Validation Accuracy')
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Accuracy')
    matplotlib.pyplot.legend()

    matplotlib.pyplot.subplot(1, 2, 2)
    matplotlib.pyplot.plot(epochs, train_losses, label='Training Loss')
    matplotlib.pyplot.plot(epochs, val_losses, label='Validation Loss')
    matplotlib.pyplot.title('Training y Validation Loss')
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Loss')
    matplotlib.pyplot.legend()

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

    # get true labels and predictions from test data
    true_labels, predictions = evaluate_model(model, test_dataloader, device)

    # evaluate model with metrics
    print("\nTest Accuracy:", accuracy_score(true_labels, predictions))
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))

    # compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    # plot ROC curve and AUC value
    matplotlib.pyplot.figure(figsize=(8, 6))
    matplotlib.pyplot.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'AUC = {roc_auc:.2f}'
        )
    matplotlib.pyplot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    matplotlib.pyplot.xlabel('Ratio de Falsos Positivos')
    matplotlib.pyplot.ylabel('Ratio de Verdaderos Positivos')
    matplotlib.pyplot.title('Curva "Receiver Operating Characteristic" (ROC)')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
