# Multilingual Toxicity Classifier

This repository houses a machine learning project focused on multilingual text classification. The goal of the project is to develop a model capable of classifying text into two categories: "Toxic" and "Non-toxic." The project involves text data in three languages: Spanish, English, and French.

## Data Preparation
After pre-processing, the dataset consists of 9921 rows for each language, with an equal distribution of samples across languages. To handle the multilingual nature of the project, a strategy of concatenating all text data for training has been employed. This involves combining texts from all three languages, along with their corresponding labels.

## Model Training
A BERT-type model is fine-tuned for text classification, in particular, the `bert-base-multilingual-cased` (https://huggingface.co/bert-base-multilingual-cased). The training process involves tokenizing the combined multilingual texts and using them to train the model. Class imbalance, where "Class 0" has significantly more examples than "Class 1," is addressed through class weight balancing.

## Class Weight Balancing
The class weights are calculated based on the inverse of the class frequencies in the training set. This approach helps mitigate the impact of class imbalance during training and improves the model's ability to generalize to minority classes.

## Training and Evaluation
The model is trained over multiple epochs, with periodic validation to monitor its performance. Despite class weight balancing, the evaluation on the test set reveals challenges in predicting "Class 1", where the model struggles to achieve meaningful results.

## Future Considerations
To improve the model's performance on minority classes, further strategies such as experimenting with different neural network architectures, fine-tuning hyperparameters, or exploring advanced techniques like data augmentation could be considered.

## Repository Structure
This repository contains a Jupyter Notebook with the sequential exploration of the data, the model training and its evaluation. It also contains a folder `data` with the relevant data used and created in the project. Finally, it also contains a folder `src` with the Python source of the project, along with some relevant unitary tests.
