import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
from model import Net

class FederatedLearning:
    def __init__(self, num_clients, epochs, path='./cifar_net.pth', max_workers=5, batch_size=4):
        self.num_clients = num_clients
        self.epochs = epochs
        self.path = path
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    def train_local_model(self, client_id, client_loader, net, global_progress):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(self.epochs):
            running_loss = 0.0
            progress_bar = tqdm(enumerate(client_loader, 0), total=len(client_loader), desc=f'Client {client_id+1}, Epoch {epoch+1}', leave=False)
            for i, data in progress_bar:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (i + 1))

            global_progress.update(1)

    def train_client(self, client_id, client_loader, global_model_state_dict, results, global_progress):
        try:
            local_model = Net().to(self.device)
            local_model.load_state_dict(global_model_state_dict)
            self.train_local_model(client_id, client_loader, local_model, global_progress)
            results[client_id] = local_model.state_dict()
        except Exception as e:
            logging.error(f"Error in client {client_id}", exc_info=True)

    def federated_averaging(self, global_model, client_models):
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.mean(torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))]), dim=0)
        global_model.load_state_dict(global_dict)

    def train_federated(self, trainset):
        from data import DataLoaderWrapper
        client_datasets = DataLoaderWrapper.split_dataset(trainset, self.num_clients)
        global_model = Net().to(self.device)

        total_steps = self.num_clients * self.epochs
        with tqdm(total=total_steps, desc="Federated Learning Progress") as global_progress:
            for epoch in range(self.epochs):
                client_loaders = [DataLoader(client_datasets[client_id], batch_size=self.batch_size, shuffle=True, num_workers=2) for client_id in range(self.num_clients)]
                client_models_state_dict = global_model.state_dict()
                results = [None] * self.num_clients

                # Use ThreadPoolExecutor for parallel execution
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self.train_client, client_id, client_loaders[client_id], client_models_state_dict, results, global_progress) for client_id in range(self.num_clients)]
                    for future in futures:
                        future.result()  # Wait for all threads to complete

                client_models = [Net().to(self.device) for _ in range(self.num_clients)]
                for client_id, state_dict in enumerate(results):
                    client_models[client_id].load_state_dict(state_dict)

                self.federated_averaging(global_model, client_models)

        torch.save(global_model.state_dict(), self.path)
        print('\nFinished Training\n')

    def check(self, testloader):
        net = Net().to(self.device)
        net.load_state_dict(torch.load(self.path))
        
        correct = 0
        total = 0
        with tqdm(total=len(testloader), desc="Testing Progress") as progress_bar:
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    progress_bar.update(1)

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
