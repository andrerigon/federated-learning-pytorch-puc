import enum
import json
import logging
import random
from typing import TypedDict
from collections import OrderedDict
import base64
import json
from io import BytesIO
from io import BytesIO
import logging
import gzip
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.plugin.mission_mobility import MissionMobilityPlugin, MissionMobilityConfiguration, LoopMission

from image_classifier_autoencoders import download_dataset, split_dataset, Autoencoder, device
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import threading
import torch


class SimpleSender(enum.Enum):
    SENSOR = 0
    UAV = 1
    GROUND_STATION = 2


class SimpleMessage(TypedDict):
    packet_count: int
    sender_type: int
    sender: int
    payload: str
    type: str



def report_message(message: SimpleMessage) -> str:
    return ''
    # return (f"Received message with {message['packet_count']} packets from "
            # f"{SimpleSender(message['sender_type']).name} {message['sender']}")


class SimpleSensorProtocol(IProtocol):
    _log: logging.Logger
    packet_count: int
    remaining_energy: int

    def initialize(self) -> None:
        self.remaining_energy = random_integer = random.randint(1, 5)
        self._log = logging.getLogger()
        self.packet_count = 0
       
        self.trainset, self.testset = download_dataset()
        self.splited_dataset = split_dataset(self.trainset, 4)
        self.id = self.provider.get_id()

        self.global_model = Autoencoder().to(device)
        self.loader = DataLoader(self.splited_dataset[self.id], batch_size=4, shuffle=True, num_workers=2)

        self.thread = threading.Thread(target=self.start_training)
        self.finished = False
        self.model_updated = False
        self.thread.start()
        

    def start_training(self):
        while not self.finished:
            self.train()    

    

    def train(self, epochs = 1):
        # Specify the quantization engine
        torch.backends.quantized.engine = 'qnnpack'  # Use 'qnnpack' for ARM architectures
        try:
            local_model = Autoencoder().to(device)
            local_model.load_state_dict(self.global_model.state_dict())
            local_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(local_model, inplace=True)

            criterion = nn.MSELoss()
            optimizer = optim.AdamW(self.global_model.parameters(), lr=0.0001, weight_decay=1e-5)  # Reduced learning rate
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            for epoch in range(epochs):
                running_loss = 0.0
                progress_bar = tqdm(enumerate(self.loader, 0), total=len(self.loader), desc=f'Client {self.id+1}, Epoch {epoch+1}', leave=False)
                for i, data in progress_bar:
                    inputs = data[0].to(device)

                    optimizer.zero_grad()

                    outputs = self.global_model(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    progress_bar.set_postfix(loss=running_loss / (i + 1))
                scheduler.step()
            local_model.eval()
            local_model = torch.quantization.convert(local_model)
            self.current_state = local_model.state_dict()   
            self.model_updated = True 

        except Exception as e:
            logging.error(f"Error in client {self.id}", exc_info=True)    

    # def _generate_packet(self) -> None:
    #     if self.remaining_energy <= 0:
    #         self._log.info(f"Device energy level is low. Won't send packages")
    #         return 
    

    #     if self.model_updated: 
    #         self.packet_count += 1
    #         # self._log.info(f"Trained concluded packet, current count {self.packet_count}")
    #         self.provider.schedule_timer("", self.provider.current_time() + 10000)
    #         self.remaining_energy -= 1

    def handle_timer(self, timer: str) -> None:
        self._generate_packet()

    def serialize_state_dict(self, state_dict):
        # Serialize the entire state_dict to a byte stream
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        
        # Compress the byte stream
        compressed_buffer = BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as f:
            f.write(buffer.getvalue())
        
        compressed_data = compressed_buffer.getvalue()
        
        # Encode the compressed data to base64
        compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
        
        # Log the size of the compressed data
        logging.info(f"Serialized and compressed state_dict size: {len(compressed_data)} bytes")
        
        return json.dumps(compressed_base64)

    def handle_packet(self, message: str) -> None:
        simple_message: SimpleMessage = json.loads(message)
        # self._log.info(report_message(simple_message))
        # self._log.info(f'sensor received from sender: {simple_message['sender_type']} and mode updated is {self.model_updated}')
        if simple_message['sender_type'] == SimpleSender.UAV.value and self.model_updated:
            
            self.model_updated = False
            # response: SimpleMessage = {
            #     'packet_count': self.packet_count,
            #     'sender_type': SimpleSender.SENSOR.value,
            #     'sender': self.provider.get_id()
            # }
            response: SimpleMessage = {
                'payload': self.serialize_state_dict(self.current_state),
                'sender': self.id,
                'packet_count': self.packet_count,
                'type': 'model_update'
            }


            command = SendMessageCommand(json.dumps(response), simple_message['sender'])
            self.provider.send_communication_command(command)

            # self._log.info(f"Sent {response['packet_count']} packets to UAV {simple_message['sender']}")

            self.packet_count += 1
        if simple_message['type'] == 'model_update':
            self._log.info(f"Got model update from UVA")
            state = decompress_and_deserialize_state_dict(simple_message['payload'])
            self.global_model.load_state_dict(state)


    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        self.finished = True
        self._log.info(f"Final packet count: {self.packet_count}")


mission_list = [
    [
        (0, 0, 20),
        (150, 0, 20)
    ],
    [
        (0, 0, 20),
        (0, 150, 20)
    ],
    [
        (0, 0, 20),
        (-150, 0, 20)
    ],
    [
        (0, 0, 20),
        (0, -150, 20)
    ]
]


def decompress_and_deserialize_state_dict(serialized_state_dict):
    # Decode the base64 to get compressed byte stream
    compressed_data = base64.b64decode(serialized_state_dict.encode('utf-8'))
    
    # Decompress the byte stream
    compressed_buffer = BytesIO(compressed_data)
    with gzip.GzipFile(fileobj=compressed_buffer, mode='rb') as f:
        buffer = BytesIO(f.read())
    
    buffer.seek(0)
    
    # Deserialize the state_dict
    return torch.load(buffer)

class SimpleUAVProtocol(IProtocol):
    _log: logging.Logger

    packet_count: int

    _mission: MissionMobilityPlugin

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.packet_count = 0

        self.id = self.provider.get_id()
        self._mission = MissionMobilityPlugin(self, MissionMobilityConfiguration(
            loop_mission=LoopMission.REVERSE,
        ))

        self._mission.start_mission(mission_list.pop())

        self.model = Autoencoder().to(device)

        self._send_heartbeat()

    def serialize_state_dict(self, state_dict):
        # Serialize the entire state_dict to a byte stream
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        
        # Compress the byte stream
        compressed_buffer = BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as f:
            f.write(buffer.getvalue())
        
        compressed_data = compressed_buffer.getvalue()
        
        # Encode the compressed data to base64
        compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
        
        # Log the size of the compressed data
        logging.info(f"Serialized and compressed state_dict size: {len(compressed_data)} bytes")
        
        return json.dumps(compressed_base64)    

    def _send_heartbeat(self) -> None:
        # self._log.info(f"Sending heartbeat, current count {self.packet_count}")

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
        self._log.info(f'got message with size: {len(message)}')
        message: SimpleMessage = json.loads(message)

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
        
        # simple_message: SimpleMessage = json.loads(message)
        # self._log.info(report_message(simple_message))

        # if simple_message['sender_type'] == SimpleSender.SENSOR.value:
        #     self.packet_count += simple_message['packet_count']
        #     # self._log.info(f"Received {simple_message['packet_count']} packets from "
        #                 #    f"sensor {simple_message['sender']}. Current count {self.packet_count}")
        # elif simple_message['sender_type'] == SimpleSender.GROUND_STATION.value:
        #     # self._log.info("Received acknowledgment from ground station")
        #     self.packet_count = 0

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        _, testset = download_dataset()
        testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        self.check(testloader)
        self._log.info(f"Final packet count: {self.packet_count}")

    def check(self, testloader):
        
        criterion = nn.MSELoss()
        total_loss = 0
        with tqdm(total=len(testloader), desc="Testing Progress") as progress_bar:
            with torch.no_grad():
                for data in testloader:
                    images = data[0].to(device)
                    outputs = self.model(images)
                    loss = criterion(outputs, images)
                    total_loss += loss.item()
                    progress_bar.update(1)

        print(f'Mean Squared Error of the network on the test images: {total_loss / len(testloader)}')

        # Extract features and evaluate clustering
        features, labels = extract_features(testloader, self.model)
        evaluate_clustering(features, labels)

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

class SimpleGroundStationProtocol(IProtocol):
    _log: logging.Logger
    packet_count: int

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.packet_count = 0

    def handle_timer(self, timer: str) -> None:
        pass

    def handle_packet(self, message: str) -> None:
        simple_message: SimpleMessage = json.loads(message)
        self._log.info(report_message(simple_message))

        if simple_message['sender_type'] == SimpleSender.UAV.value:
            response: SimpleMessage = {
                'packet_count': self.packet_count,
                'sender_type': SimpleSender.GROUND_STATION.value,
                'sender': self.provider.get_id()
            }

            command = SendMessageCommand(json.dumps(response), simple_message['sender'])
            self.provider.send_communication_command(command)

            self.packet_count += simple_message['packet_count']
            # self._log.info(f"Sent acknowledgment to UAV {simple_message['sender']}. Current count {self.packet_count}")

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        self._log.info(f"Final packet count: {self.packet_count}")
