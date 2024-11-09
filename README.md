# Covid19_training_with_inference

This repository contains code for training and evaluating deep learning models with various hyperparameter configurations. The project allows you to experiment with different optimizers, learning rates, batch sizes, and weight decay strategies to optimize your model's performance. The goal is to provide a framework for running multiple training configurations and comparing their results, including metrics such as training loss, validation accuracy, and training time.

Table of Contents
Project Overview
Installation
Usage
Hyperparameters
Results
License
Acknowledgments
Project Overview
This repository includes scripts for:

Model Training: Train deep learning models using different configurations and optimizers.
Hyperparameter Tuning: Evaluate multiple configurations of hyperparameters such as learning rate, batch size, and optimizer choice (Adam, SGD, AdamW, RMSProp, Adagrad).
Model Evaluation: Track and plot training and validation loss/accuracy over epochs.
Training Time Logging: Log the time taken for each epoch, helping to identify bottlenecks in training.
The model is trained on a classification task using the PyTorch framework.

Installation
To get started with this project, you'll need to have Python 3.x installed along with the necessary dependencies. You can install the required packages using pip.

Clone this repository
bash
Copy code
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install the dependencies
bash
Copy code
pip install -r requirements.txt
Make sure you have the following dependencies in your requirements.txt:

makefile
Copy code
torch==<version>
matplotlib==<version>
numpy==<version>
For GPU support, you'll need to install the appropriate version of PyTorch with CUDA support. You can find instructions for this on the official PyTorch website.

Usage
1. Model Setup:
The model is initialized and trained using different configurations specified in the configs list. You can modify this list to try different combinations of learning rates, batch sizes, and optimizers.

python
Copy code
configs = [
    {'device': 'cpu', 'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'adam', 'weight_decay': 0},
    {'device': 'cuda', 'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'adam', 'weight_decay': 0},
    # More configurations...
]
2. Running the Experiment:
Run the main script to train the model with the defined configurations and plot the results:

bash
Copy code
python main.py
This will run the training for each configuration and generate plots for the following metrics:

Training and validation loss
Training and validation accuracy
Epoch training time
3. Evaluation:
After training, the results are stored and compared. You will see plots comparing the performance of different configurations.

4. Plotting Results:
The results are plotted using matplotlib. The plots will show:

Loss: Comparison of training and validation loss over epochs.
Accuracy: Comparison of training and validation accuracy over epochs.
Epoch Time: Time taken for each epoch.
Hyperparameters
The hyperparameters for training are defined in the configs list. Here are some of the key hyperparameters you can modify:

device: Whether to use the CPU or CUDA (GPU).
learning_rate: Learning rate for the optimizer.
batch_size: The number of samples per gradient update.
optimizer: The optimizer used for training (e.g., 'adam', 'sgd', 'adamw', 'rmsprop', 'adagrad').
momentum: Momentum for optimizers like SGD and RMSprop.
weight_decay: L2 regularization strength for the optimizer.
Example Configuration:
python
Copy code
{
    'device': 'cuda',
    'learning_rate': 0.001,
    'batch_size': 64,
    'optimizer': 'adamw',
    'weight_decay': 0.1
}
Results
After running the training with different configurations, the results will include:

Training Loss: The loss function value calculated on the training set.
Validation Loss: The loss function value calculated on the validation set.
Training Accuracy: The accuracy of the model on the training set.
Validation Accuracy: The accuracy of the model on the validation set.
Epoch Training Time: The time taken to complete each epoch.
The results for each configuration are plotted and can be compared to identify the optimal hyperparameters for the task at hand.
