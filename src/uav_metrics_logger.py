import enum
import json
import logging
import random
import threading
import queue
from typing import TypedDict
from collections import OrderedDict
import base64
import gzip
from io import BytesIO
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import gc
import torch.multiprocessing as mp
import multiprocessing.shared_memory as shm


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.plugin.mission_mobility import MissionMobilityPlugin, MissionMobilityConfiguration, LoopMission
from image_classifier_autoencoders import download_dataset, split_dataset, Autoencoder, device as ae_device
from image_classifier_supervisioned import SupervisedModel, device as sup_device
from sensor_protocol import SimpleMessage, SimpleSender


def plot_mse(mse_values, output_dir):
    plt.figure()
    plt.plot(mse_values, marker='o')
    plt.title('Mean Squared Error over Test Batches', fontsize=14)
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mse_plot.png'))
    plt.close()

def plot_clustering_metrics(accuracy, ari, output_dir):
    metrics = {'Clustering Accuracy': accuracy, 'Adjusted Rand Index': ari}
    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Clustering Evaluation Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'clustering_metrics.png'))
    plt.close()

def plot_confusion_matrix(labels, predicted_labels, classes, output_dir):
    cm = confusion_matrix(labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_tsne(features, labels, output_dir):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_features = tsne.fit_transform(features)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=range(10))
    plt.title('t-SNE Visualization of Features', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'tsne_plot.png'))
    plt.close()

class UAVMetricsLogger:
    def __init__(self, id, output_dir):
        self.id = id
        self.output_dir = output_dir
        self.training_cycles = {}
        self.model_updates = {}
        self.success_rates = {}
        self.staleness_records = []    # To record staleness metrics

    def record_staleness(self, sender_id, staleness, timestamp):
        self.staleness_records.append({
            'sensor_id': sender_id,
            'staleness': staleness,
            'timestamp': timestamp
        })

    def update(self, sender_id, training_cycles, model_updates, success_rates):
        self.training_cycles[sender_id] = training_cycles
        self.model_updates[sender_id] = model_updates
        self.success_rates[sender_id] = success_rates

    def record_staleness_metrics(self):
        if not self.staleness_records:
            return
        
        staleness_values = [record['staleness'] for record in self.staleness_records]
        average_staleness = sum(staleness_values) / len(staleness_values)
        max_staleness = max(staleness_values)
        min_staleness = min(staleness_values)
        print(f"Average Staleness: {average_staleness}")
        print(f"Max Staleness: {max_staleness}")
        print(f"Min Staleness: {min_staleness}")

        with open(os.path.join(self.output_dir, f'staleness_metrics_{self.id}.txt'), 'w') as f:
            f.write(f"Average Staleness: {average_staleness}\n")
            f.write(f"Max Staleness: {max_staleness}\n")
            f.write(f"Min Staleness: {min_staleness}\n")
            f.write("Staleness Records:\n")
            for record in self.staleness_records:
                f.write(f"{record}\n")  

        staleness_df = pd.DataFrame(self.staleness_records)
        staleness_df.to_csv(os.path.join(self.output_dir, f'staleness_records_{self.id}.csv'), index=False)

        plt.figure()
        for sensor_id in staleness_df['sensor_id'].unique():
            sensor_data = staleness_df[staleness_df['sensor_id'] == sensor_id]
            plt.plot(sensor_data['timestamp'], sensor_data['staleness'], label=f"Sensor {sensor_id}")
        plt.xlabel('Timestamp', fontsize=14)
        plt.ylabel('Staleness', fontsize=14)
        plt.title(f'Staleness Over Time for UAV {self.id}', fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'staleness_over_time_{self.id}.png'))
        plt.close()                      
