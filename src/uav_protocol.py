import enum
import json
import logging
import random
from typing import TypedDict
from collections import OrderedDict
import base64
import gzip
from io import BytesIO
import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.plugin.mission_mobility import MissionMobilityPlugin, MissionMobilityConfiguration, LoopMission
from image_classifier_autoencoders import download_dataset, split_dataset, Autoencoder, device
from sensor_protocol import SimpleMessage, SimpleSender

def report_message(message: SimpleMessage) -> str:
    return ''

mission_list = [
    
        (0, 0, 20),
        (150, 0, 20)
    ,
    
        (0, 0, 20),
        (0, 150, 20)
    ,
    
        (0, 0, 20),
        (-150, 0, 20)
    ,
    
        (0, 0, 20),
        (0, -150, 20)
    ,
    
        (0, 0, 20),
        (150, 0, 20)
    ,
    
        (0, 0, 20),
        (0, 150, 20)
    ,
    
        (0, 0, 20),
        (-150, 0, 20)
    ,
    
        (0, 0, 20),
        (0, -150, 20)
    

]


def decompress_and_deserialize_state_dict(serialized_state_dict):
    compressed_data = base64.b64decode(serialized_state_dict.encode('utf-8'))
    compressed_buffer = BytesIO(compressed_data)
    with gzip.GzipFile(fileobj=compressed_buffer, mode='rb') as f:
        buffer = BytesIO(f.read())
    buffer.seek(0)
    return torch.load(buffer)

def plot_mse(mse_values, output_dir):
    plt.figure()
    plt.plot(mse_values, marker='o')
    plt.title('Mean Squared Error over Test Batches')
    plt.xlabel('Batch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mse_plot.png'))
    plt.close()

def plot_clustering_metrics(accuracy, ari, output_dir):
    metrics = {'Clustering Accuracy': accuracy, 'Adjusted Rand Index': ari}
    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Clustering Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'clustering_metrics.png'))
    plt.close()

def plot_confusion_matrix(labels, predicted_labels, classes, output_dir):
    cm = confusion_matrix(labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_tsne(features, labels, output_dir):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_features = tsne.fit_transform(features)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=range(10))
    plt.title('t-SNE Visualization of Features')
    plt.savefig(os.path.join(output_dir, 'tsne_plot.png'))
    plt.close()

class SimpleUAVProtocol(IProtocol):
    _log: logging.Logger
    packet_count: int
    _mission: MissionMobilityPlugin

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.packet_count = 0
        self.id = self.provider.get_id()
        self._mission = MissionMobilityPlugin(self, MissionMobilityConfiguration(loop_mission=LoopMission.REVERSE))
        self._mission.start_mission(mission_list)
        self.model = self.load_model()
        self._send_heartbeat()
        self.training_cycles = {}
        self.model_updates = {}

    def load_model(self):
        model_path = self.get_last_model_path()
        model = Autoencoder().to(device)
        if model_path:
            model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        return model

    def get_last_model_path(self):
        output_base_dir = 'output'
        if not os.path.exists(output_base_dir):
            return None
        dirs = sorted([d for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))], reverse=True)
        if not dirs:
            return None
        latest_dir = os.path.join(output_base_dir, dirs[0])
        model_path = os.path.join(latest_dir, 'model.pth')
        return model_path if os.path.exists(model_path) else None    

    def serialize_state_dict(self, state_dict):
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        compressed_buffer = BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as f:
            f.write(buffer.getvalue())
        compressed_data = compressed_buffer.getvalue()
        compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
        logging.info(f"Serialized and compressed state_dict size: {len(compressed_data)} bytes")
        return json.dumps(compressed_base64)    

    def _send_heartbeat(self) -> None:
        message: SimpleMessage = {
            'packet_count': self.packet_count,
            'sender_type': SimpleSender.UAV.value,
            'sender': self.provider.get_id(),
            'type': 'ping'
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)
        self.provider.schedule_timer("", self.provider.current_time() + 1)

    def update_global_model_with_client(self, client_dict, alpha=0.1):
        global_dict = self.model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = (1 - alpha) * global_dict[k] + alpha * client_dict[k].dequantize()
        self.model.load_state_dict(global_dict)
        self.model.eval()
        self.model = torch.quantization.convert(self.model)

    def handle_timer(self, timer: str) -> None:
        self._send_heartbeat()

    def handle_packet(self, message: str) -> None:
        # self._log.info(f'got message with size: {len(message)}')
        message: SimpleMessage = json.loads(message)
        self.training_cycles[message['sender']] = message['training_cycles']
        self.model_updates[message['sender']] = message['model_updates']
        decompressed_state_dict = decompress_and_deserialize_state_dict(message['payload'])
        self.update_global_model_with_client(decompressed_state_dict)
        self._log.info(f'Sending model update')
        message: SimpleMessage = {
            'sender_type': SimpleSender.UAV.value,
            'payload': self.serialize_state_dict(self.model.state_dict()),
            'sender': self.id,
            'packet_count': self.packet_count,
            'type': 'model_update'
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        _, testset = download_dataset()
        testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        output_dir = os.path.join('output', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        self.check(testloader, output_dir)
        self._log.info(f"Final packet count: {self.packet_count}")

    def check(self, testloader, output_dir):
        criterion = nn.MSELoss()
        total_loss = 0
        mse_values = []
        with tqdm(total=len(testloader), desc="Testing Progress") as progress_bar:
            with torch.no_grad():
                for data in testloader:
                    images = data[0].to(device)
                    outputs = self.model(images)
                    loss = criterion(outputs, images)
                    total_loss += loss.item()
                    mse_values.append(loss.item())
                    progress_bar.update(1)

        mean_mse = total_loss / len(testloader)
        print(f'Mean Squared Error of the network on the test images: {mean_mse}')
        plot_mse(mse_values, output_dir)

        features, labels = extract_features(testloader, self.model)
        accuracy, ari = evaluate_clustering(features, labels, output_dir)
        plot_clustering_metrics(accuracy, ari, output_dir)

        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pth'))
        with open(os.path.join(output_dir, 'stats.txt'), 'w') as f:
            f.write(f'Mean Squared Error: {mean_mse}\n')
            f.write(f'Clustering Accuracy: {accuracy}\n')
            f.write(f'Adjusted Rand Index: {ari}\n')
            f.write(f'Model Updates: {self.model_updates}\n')
            f.write(f'Training Cycles: {self.training_cycles}\n')

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

def evaluate_clustering(features, labels, output_dir, n_clusters=10):
    n_clusters = min(n_clusters, len(features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    predicted_labels = kmeans.labels_
    label_mapping = {}
    for cluster in range(n_clusters):
        cluster_indices = np.where(predicted_labels == cluster)[0]
        true_labels = labels[cluster_indices]
        most_common_label = np.bincount(true_labels).argmax()
        label_mapping[cluster] = most_common_label
    mapped_predicted_labels = np.vectorize(label_mapping.get)(predicted_labels)
    accuracy = accuracy_score(labels, mapped_predicted_labels)
    ari = adjusted_rand_score(labels, mapped_predicted_labels)
    print(f'Clustering Accuracy: {accuracy}')
    print(f'Adjusted Rand Index: {ari}')
    plot_confusion_matrix(labels, mapped_predicted_labels, classes=range(10), output_dir=output_dir)
    plot_tsne(features, labels, output_dir)
    return accuracy, ari