import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import sys
import multiprocessing as mp
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

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
PATH = './autoencoder.pth'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simplified Autoencoder model definition
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # b, 32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 8, 8
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # b, 64, 8, 8
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # b, 64, 8, 8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 128, 4, 4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # b, 256, 2, 2
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # b, 512, 1, 1
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # b, 256, 2, 2
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # b, 128, 4, 4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # b, 64, 8, 8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # b, 32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # b, 3, 32, 32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-5)  # Reduced learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(client_loader, 0), total=len(client_loader), desc=f'Client {client_id+1}, Epoch {epoch+1}', leave=False)
        for i, data in progress_bar:
            inputs = data[0].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))
        scheduler.step()

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
        local_model = Autoencoder().to(device)
        local_model.load_state_dict(global_model_state_dict)
        train_local_model(client_id, client_loader, local_model, epochs, global_progress)
        results[client_id] = local_model.state_dict()
    except Exception as e:
        logging.error(f"Error in client {client_id}", exc_info=True)

# Function to orchestrate federated learning across all clients
def train_federated(trainset, num_clients, epochs, path=PATH, max_workers=5, batch_size=4):
    client_datasets = split_dataset(trainset, num_clients)
    global_model = Autoencoder().to(device)

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

            client_models = [Autoencoder().to(device) for _ in range(num_clients)]
            for client_id, state_dict in enumerate(results):
                client_models[client_id].load_state_dict(state_dict)

            federated_averaging(global_model, client_models)

    torch.save(global_model.state_dict(), path)
    print('\nFinished Training\n')

# Function to extract features using the encoder part of the autoencoder
def extract_features(dataloader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            images, targets = data[0].to(device), data[1]
            encoded = model.encoder(images)
            features.append(encoded.view(encoded.size(0), -1).cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

# Function to apply K-means and evaluate clustering performance
def evaluate_clustering(features, labels, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    predicted_labels = kmeans.labels_
    # Map the predicted labels to the most frequent true labels in each cluster
    label_mapping = {}
    for cluster in range(n_clusters):
        cluster_indices = np.where(predicted_labels == cluster)[0]
        true_labels = labels[cluster_indices]
        most_common_label = np.bincount(true_labels).argmax()
        label_mapping[cluster] = most_common_label
    
    # Replace predicted labels with mapped labels
    mapped_predicted_labels = np.vectorize(label_mapping.get)(predicted_labels)
    
    accuracy = accuracy_score(labels, mapped_predicted_labels)
    ari = adjusted_rand_score(labels, mapped_predicted_labels)
    print(f'Clustering Accuracy: {accuracy}')
    print(f'Adjusted Rand Index: {ari}')

# Function to evaluate the global model on the test set using K-means clustering
def check(testloader, path=PATH):
    net = Autoencoder().to(device)
    net.load_state_dict(torch.load(path))
    
    criterion = nn.MSELoss()
    total_loss = 0
    with tqdm(total=len(testloader), desc="Testing Progress") as progress_bar:
        with torch.no_grad():
            for data in testloader:
                images = data[0].to(device)
                outputs = net(images)
                loss = criterion(outputs, images)
                total_loss += loss.item()
                progress_bar.update(1)

    print(f'Mean Squared Error of the network on the test images: {total_loss / len(testloader)}')

    # Extract features and evaluate clustering
    features, labels = extract_features(testloader, net)
    evaluate_clustering(features, labels)

# Main function to execute the federated learning workflow
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Federated Learning with Autoencoders on CIFAR-10')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')  # Increased epochs
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
