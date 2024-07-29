import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import sys
import multiprocessing as mp
import argparse

# Set up logging to file and stdout
logging.basicConfig(filename='stderr.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Function to handle uncaught exceptions and log them
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# Set up logging for warnings
logging.captureWarnings(True)

# Model save path
PATH = './cifar_net.pth'

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Device configuration
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# Neural network model definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 45, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(45, 66, 5)
        self.fc1 = nn.Linear(66 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 104)
        self.fc3 = nn.Linear(104, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to display images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Function to download and transform CIFAR-10 dataset
def download_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
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
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
        local_model = Net().to(device)
        local_model.load_state_dict(global_model_state_dict)
        train_local_model(client_id, client_loader, local_model, epochs, global_progress)
        results[client_id] = local_model.state_dict()
    except Exception as e:
        logging.error(f"Error in client {client_id}", exc_info=True)

# Function to orchestrate federated learning across all clients
def train_federated(trainset, num_clients, epochs, path=PATH, max_workers=5, batch_size=4):
    client_datasets = split_dataset(trainset, num_clients)
    global_model = Net().to(device)

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

            client_models = [Net().to(device) for _ in range(num_clients)]
            for client_id, state_dict in enumerate(results):
                client_models[client_id].load_state_dict(state_dict)

            federated_averaging(global_model, client_models)

    torch.save(global_model.state_dict(), path)
    print('\nFinished Training\n')

# Function to evaluate the global model on the test set
def check(testloader, path=PATH):
    net = Net().to(device)
    net.load_state_dict(torch.load(path))
    
    correct = 0
    total = 0
    with tqdm(total=len(testloader), desc="Testing Progress") as progress_bar:
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                progress_bar.update(1)

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# Main function to execute the federated learning workflow
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Federated Learning with PyTorch on CIFAR-10')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')

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
