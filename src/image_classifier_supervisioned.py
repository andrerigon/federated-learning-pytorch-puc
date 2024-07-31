import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
import torch.nn.functional as F
import numpy as np
import argparse
import multiprocessing as mp


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple CNN model for supervised classification
class SupervisedModel(nn.Module):
    def __init__(self):
        super(SupervisedModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # Assuming input images are 32x32
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)  # CIFAR-10 has 10 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x


# Function to download and transform CIFAR-10 dataset
def download_dataset():
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    return trainset, testset

# Function to split dataset into subsets for each client
def split_dataset(dataset, num_clients):
    data_indices = list(range(len(dataset)))
    split_indices = np.array_split(data_indices, num_clients)
    client_datasets = [Subset(dataset, indices) for indices in split_indices]
    return client_datasets

# Function to train a local model on a client's data
def train_local_model(client_id, client_loader, net, epochs, global_progress):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(client_loader, 0), total=len(client_loader), desc=f'Client {client_id+1}, Epoch {epoch+1}', leave=False)
        for i, data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        global_progress.update(1)

# Function to average the model parameters from all clients
def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.mean(torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))]), dim=0)
    global_model.load_state_dict(global_dict)

# Function to train a client's model in parallel
def train_client(client_id, client_loader, global_model_state_dict, results, epochs, global_progress):
    try:
        local_model = SupervisedModel().to(device)
        local_model.load_state_dict(global_model_state_dict)
        train_local_model(client_id, client_loader, local_model, epochs, global_progress)
        results[client_id] = local_model.state_dict()
    except Exception as e:
        logging.error(f"Error in client {client_id}", exc_info=True)

# Function to orchestrate federated learning across all clients
def train_federated(trainset, num_clients, epochs, path='./cnn.pth', max_workers=5, batch_size=4):
    client_datasets = split_dataset(trainset, num_clients)
    global_model = SupervisedModel().to(device)

    total_steps = num_clients * epochs
    with tqdm(total=total_steps, desc="Federated Learning Progress") as global_progress:
        for epoch in range(epochs):
            client_loaders = [DataLoader(client_datasets[client_id], batch_size=batch_size, shuffle=True, num_workers=2) for client_id in range(num_clients)]
            client_models_state_dict = global_model.state_dict()
            results = [None] * num_clients

            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(train_client, client_id, client_loaders[client_id], client_models_state_dict, results, 1, global_progress) for client_id in range(num_clients)]
                for future in futures:
                    future.result()  # Wait for all threads to complete

            client_models = [SupervisedModel().to(device) for _ in range(num_clients)]
            for client_id, state_dict in enumerate(results):
                client_models[client_id].load_state_dict(state_dict)

            federated_averaging(global_model, client_models)

    torch.save(global_model.state_dict(), path)
    print('\nFinished Training\n')

# Function to evaluate the global model on the test set
def check(testloader, path='./cnn.pth'):
    net = SupervisedModel().to(device)
    net.load_state_dict(torch.load(path))
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with tqdm(total=len(testloader), desc="Testing Progress") as progress_bar:
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
                progress_bar.update(1)

    accuracy = 100 * correct / total
    mean_loss = total_loss / len(testloader)
    cm = confusion_matrix(all_labels, all_predicted)
    print(f'Mean Loss: {mean_loss}')
    print(f'Accuracy: {accuracy}%')
    print(f'Confusion Matrix:\n{cm}')

# Main function to execute the federated learning workflow
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Federated Learning with Supervised Learning on CIFAR-10')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of workers')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')

    args = parser.parse_args()

    try:
        trainset, testset = download_dataset()
        train_federated(trainset, args.num_clients, args.epochs, max_workers=args.max_workers, batch_size=args.batch_size)
        
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        check(testloader)
    except Exception as e:
        logging.error("Error in main function", exc_info=True)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
