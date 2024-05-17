
# Federated Learning with PyTorch on CIFAR-10

## Project Overview

This project implements a federated learning framework using PyTorch on the CIFAR-10 dataset. The goal is to simulate a scenario where multiple clients (e.g., devices in a distributed system) collaboratively train a global neural network model without sharing their local data. This approach enhances data privacy and security while leveraging decentralized data sources.

## Technologies and Techniques Involved

- **Federated Learning**: A machine learning technique where multiple decentralized devices collaboratively train a model without sharing raw data.
- **PyTorch**: An open-source machine learning library used for training the neural network model.
- **CIFAR-10**: A widely-used dataset for image classification tasks.
- **Multithreading and Parallel Processing**: Techniques to handle concurrent training of client models.
- **Logging and Exception Handling**: To ensure robust and traceable code execution.

## Requirements

- Python 3.8 or higher
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- TQDM

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/federated-learning-pytorch.git
    cd federated-learning-pytorch
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Execution

1. **Download and preprocess the CIFAR-10 dataset**.
2. **Train the federated learning model**:
    ```sh
    python src/main.py --num_clients 10 --epochs 20 --max_workers 10 --batch_size 4
    ```

3. **Evaluate the trained model on the test dataset**.

## Project Structure

```plaintext
src/
├── __init__.py
├── main.py
├── data.py
├── model.py
├── federated.py
└── utils.py
requirements.txt
README.md
```

## File Descriptions

### `src/main.py`

The main script to orchestrate the federated learning process.

- **Arguments**:
  - `--num_clients`: Number of clients to simulate.
  - `--epochs`: Number of epochs for training.
  - `--max_workers`: Maximum number of parallel workers.
  - `--batch_size`: Batch size for DataLoader.

### `src/data.py`

Handles data downloading, preprocessing, and splitting.

- **Classes and Methods**:
  - `DataLoaderWrapper.download_dataset()`: Downloads and transforms the CIFAR-10 dataset.
  - `DataLoaderWrapper.split_dataset(dataset, num_clients)`: Splits the dataset into subsets for each client.

### `src/model.py`

Defines the neural network model used for classification.

- **Classes**:
  - `Net`: A convolutional neural network with two convolutional layers and three fully connected layers.

### `src/federated.py`

Contains the logic for federated learning, including local model training and federated averaging.

- **Functions**:
  - `FederatedLearning.train_local_model(client_id, client_loader, net, epochs, global_progress)`: Trains a local model on a client's data.
  - `FederatedLearning.federated_averaging(global_model, client_models)`: Averages the model parameters from all clients.
  - `FederatedLearning.train_client(client_id, client_loader, global_model_state_dict, results, epochs, global_progress)`: Trains a client's model in parallel.
  - `FederatedLearning.train_federated(trainset, num_clients, epochs, path, max_workers, batch_size)`: Orchestrates federated learning across all clients.
  - `FederatedLearning.check(testloader, path)`: Evaluates the global model on the test set.

### `src/utils.py`

Provides utility functions for logging and exception handling.

- **Functions**:
  - `setup_logging()`: Sets up logging to file and stdout.
  - `handle_exception(exc_type, exc_value, exc_traceback)`: Handles uncaught exceptions and logs them.
  - `capture_warnings()`: Captures warnings and logs them.

## How to Use

1. **Prepare the environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Run the main script** to start the federated learning process:
    ```sh
    python src/main.py --num_clients 10 --epochs 20 --max_workers 10 --batch_size 4
    ```

3. **Monitor the training progress** through logging output and progress bars.

4. **Evaluate the trained model** using the provided test dataset.

---

