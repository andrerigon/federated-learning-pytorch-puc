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
    training_cycles: int
    model_updates: int

def report_message(message: SimpleMessage) -> str:
    return ''
    # return (f"Received message with {message['packet_count']} packets from "
            # f"{SimpleSender(message['sender_type']).name} {message['sender']}")


class SimpleSensorProtocol(IProtocol):
    _log: logging.Logger
    packet_count: int
    remaining_energy: int

    def initialize(self) -> None:
        self.remaining_energy = random.randint(1, 5)
        self._log = logging.getLogger()
        self.packet_count = 0
       
        self.trainset, self.testset = download_dataset()
        self.splited_dataset = split_dataset(self.trainset, 4)
        self.id = self.provider.get_id()
        self.global_model_changed = False

        self.global_model = Autoencoder().to(device)
        self.loader = DataLoader(self.splited_dataset[self.id], batch_size=4, shuffle=True, num_workers=4)
        self.training_cycles = 0
        self.model_updates = 0

        self.thread = threading.Thread(target=self.start_training)
        self.finished = False
        self.model_updated = False
        self.thread.start()
        

    def start_training(self):
        while not self.finished:
            self.train()            

    def train(self, epochs=1):
        # Specify the quantization engine
        torch.backends.quantized.engine = 'qnnpack'  # Use 'qnnpack' for ARM architectures
        try:
            local_model = Autoencoder().to(device)
            local_model.load_state_dict(self.global_model.state_dict())
            local_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            local_model = torch.quantization.prepare_qat(local_model, inplace=True)

            # Set anomaly detection to identify where the backward pass fails
            torch.autograd.set_detect_anomaly(True)

            criterion = nn.MSELoss()
            optimizer = optim.AdamW(self.global_model.parameters(), lr=0.0001, weight_decay=1e-5)  # Reduced learning rate
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            for epoch in range(epochs):
                if self.finished:
                    break
                running_loss = 0.0
                progress_bar = tqdm(enumerate(self.loader, 0), total=len(self.loader), desc=f'Client {self.id+1}, Epoch {epoch+1}', leave=False)
                for i, data in progress_bar:
                    if self.finished:
                        break
                    inputs = data[0].to(device)

                    if self.global_model_changed:
                        self.global_model_changed = False
                        dict = self.merge_dict(local_model.state_dict())
                        local_model = Autoencoder().to(device)
                        local_model.load_state_dict(dict)
                        local_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
                        local_model = torch.quantization.prepare_qat(local_model, inplace=True)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = local_model(inputs)
                    
                    # Compute loss
                    loss = criterion(outputs, inputs)

                    # Backward pass
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    running_loss += loss.item()
                    progress_bar.set_postfix(loss=running_loss / (i + 1))
                scheduler.step()
            
            self.training_cycles += 1

            if self.finished:
                return
            
            # Convert the model to a quantized version
            local_model.eval()
            local_model = torch.quantization.convert(local_model)
            self.current_state = local_model.state_dict()
            self.model_updated = True

        except Exception as e:
            logging.error(f"Error in client {self.id}", exc_info=True)

    def merge_dict(self, client_dict, alpha=0.1):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = (1 - alpha) * global_dict[k] + alpha * client_dict[k].dequantize()
        return global_dict

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

        self._log.info(simple_message)
        # self._log.info(f'sensor received from sender: {simple_message['sender_type']} and mode updated is {self.model_updated}')
        if simple_message['sender_type'] == SimpleSender.UAV.value and self.model_updated:
            print(f"\n\n [{self.id}] Local Model updated! \n\n")
            self.model_updated = False
            response: SimpleMessage = {
                'payload': self.serialize_state_dict(self.current_state),
                'sender': self.id,
                'packet_count': self.packet_count,
                'type': 'model_update',
                'training_cycles': self.training_cycles,
                'model_updates': self.model_updates
            }

            command = SendMessageCommand(json.dumps(response), simple_message['sender'])
            self.provider.send_communication_command(command)

            self.packet_count += 1
        if simple_message['type'] == 'model_update':
            print(f"\n\n [{self.id}] Got model update from UVA! \n\n")
            state = decompress_and_deserialize_state_dict(simple_message['payload'])
            self.global_model.load_state_dict(state)
            self.global_model_changed = True
            self.model_updates += 1

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        self.finished = True
        logging.info(f"Meg meg!")
        self._log.info(f"Final packet count: {self.packet_count}")

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