import enum
import json
import logging
import random
from typing import TypedDict
import base64
from io import BytesIO
import gzip
import torch.nn as nn
import os

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry

from image_classifier_autoencoders import download_dataset, split_dataset, Autoencoder, device as ae_device
from image_classifier_supervisioned import SupervisedModel, device as sup_device
from torch.utils.data import DataLoader
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
    success_rate: float

def report_message(message: SimpleMessage) -> str:
    return ''

class CommunicationMediator:
    def __init__(self, success_rate: float):
        self.success_rate = success_rate
        self.total_attempts = 0
        self.successful_attempts = 0

    def send_message(self, command: SendMessageCommand, provider):
        self.total_attempts += 1
        if random.random() < self.success_rate:
            self.successful_attempts += 1
            provider.send_communication_command(command)
        else:
            logging.info("Message failed to send due to simulated communication error.")

    def log_metrics(self):
        success_rate = self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0
        print(f"Message success rate: {success_rate:.2%}")
        print(f"Total attempts: {self.total_attempts}, Successful attempts: {self.successful_attempts}")

class SimpleSensorProtocol(IProtocol):
    _log: logging.Logger
    packet_count: int
    remaining_energy: int

    def __init__(self, training_mode = "autoencoder", from_scratch = False, success_rate = 1.0):
        self.training_mode = training_mode
        self.from_scratch = from_scratch
        self.success_rate = success_rate

    def initialize(self) -> None:
        self.remaining_energy = random.randint(1, 5)
        self._log = logging.getLogger()
        self.packet_count = 0
        
        self.trainset, self.testset = download_dataset()
        self.splited_dataset = split_dataset(self.trainset, 4)
        self.id = self.provider.get_id()
        self.global_model_changed = False

        self.global_model = self.load_model()
        self.loader = DataLoader(self.splited_dataset[self.id], batch_size=32, shuffle=True, num_workers=10, pin_memory=True)
        self.training_cycles = 0
        self.model_updates = 0

        # Communication Mediator with configurable success rate
        self.communicator = CommunicationMediator(success_rate = self.success_rate)

        self.thread = threading.Thread(target=self.start_training)
        self.finished = False
        self.model_updated = False
        self.thread.start()

    def load_model(self):
        model_path = self.get_last_model_path()
        if self.training_mode == 'autoencoder':
            model = Autoencoder(num_classes=10).to(ae_device)
        else:
            model = SupervisedModel().to(sup_device)

        if model_path and not self.from_scratch:
            model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        return model


    def get_last_model_path(self):
        output_base_dir = 'output'
        mode_dir = self.training_mode 
        
        if not os.path.exists(output_base_dir):
            return None
        
        dirs = sorted([d for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))], reverse=True)
        
        for d in dirs:
            latest_mode_dir = os.path.join(output_base_dir, d, mode_dir)
            if os.path.exists(latest_mode_dir):
                model_path = os.path.join(latest_mode_dir, 'model.pth')
                if os.path.exists(model_path):
                    return model_path
        
        return None   

    def start_training(self):
        while not self.finished:
            if self.training_mode == 'autoencoder':
                self.train_autoencoder()
            else:
                self.train_supervisioned()

    def train_autoencoder(self, epochs=1):
        torch.backends.quantized.engine = 'qnnpack'
        try:
            local_model = Autoencoder(num_classes=10).to(ae_device)  # Ensure num_classes matches
            local_model.load_state_dict(self.global_model.state_dict())
            local_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            local_model = torch.quantization.prepare_qat(local_model, inplace=True)

            torch.autograd.set_detect_anomaly(True)

            criterion_reconstruction = nn.MSELoss()
            criterion_classification = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(local_model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

            for epoch in range(epochs):
                if self.finished:
                    break
                running_reconstruction_loss = 0.0
                running_classification_loss = 0.0
                progress_bar = tqdm(enumerate(self.loader, 0), total=len(self.loader), desc=f'Client {self.id+1}, Epoch {epoch+1}', leave=False)
                for i, data in progress_bar:
                    if self.finished:
                        break
                    inputs, labels = data[0].to(ae_device), data[1].to(ae_device)

                    optimizer.zero_grad()

                    decoded, classified = local_model(inputs)

                    reconstruction_loss = criterion_reconstruction(decoded, inputs)
                    classification_loss = criterion_classification(classified, labels)

                    loss = reconstruction_loss + classification_loss
                    loss.backward()
                    optimizer.step()

                    running_reconstruction_loss += reconstruction_loss.item()
                    running_classification_loss += classification_loss.item()
                    progress_bar.set_postfix(reconstruction_loss=running_reconstruction_loss / (i + 1),
                                            classification_loss=running_classification_loss / (i + 1))
                scheduler.step(running_reconstruction_loss / len(self.loader))

            self.training_cycles += 1

            if self.finished:
                return

            local_model.eval()
            # Log model sizes
            self.log_model_sizes(local_model)
            local_model = torch.quantization.convert(local_model, mapping={'classifier': torch.nn.Identity})
            self.current_state = local_model.state_dict()
            self.model_updated = True

        except Exception as e:
            logging.error(f"Error in client {self.id}", exc_info=True)


    def train_supervisioned(self, epochs=1):
        torch.backends.quantized.engine = 'qnnpack'
        try:
            local_model = SupervisedModel().to(sup_device)
            local_model.load_state_dict(self.global_model.state_dict())
            local_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            local_model = torch.quantization.prepare_qat(local_model, inplace=True)
            torch.autograd.set_detect_anomaly(True)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(local_model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

            for epoch in range(epochs):
                if self.finished:
                    break
                running_loss = 0.0
                progress_bar = tqdm(enumerate(self.loader, 0), total=len(self.loader), desc=f'Client {self.id+1}, Epoch {epoch+1}', leave=False)
                for i, data in progress_bar:
                    if self.finished:
                        break
                    inputs, labels = data[0].to(sup_device), data[1].to(sup_device)

                    optimizer.zero_grad()
                    outputs = local_model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    progress_bar.set_postfix(loss=running_loss / (i + 1))
                scheduler.step(running_loss / len(self.loader))

            self.training_cycles += 1

            if self.finished:
                return

            local_model.eval()
            local_model = torch.quantization.convert(local_model, mapping={'fc_layers': torch.nn.Identity})
            self.current_state = local_model.state_dict()
            self.model_updated = True

            # Log model sizes
            self.log_model_sizes(local_model)

        except Exception as e:
            logging.error(f"Error in client {self.id}", exc_info=True)

    def log_model_sizes(self, model):
        non_quantized_size = self.get_model_size(model)
        quantized_size = self.get_model_size(torch.quantization.convert(model))
        print(f"Non-quantized model size: {non_quantized_size} bytes")
        print(f"Quantized model size: {quantized_size} bytes")

    def get_model_size(self, model):
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        return len(buffer.getvalue())
    
    def handle_timer(self, timer: str) -> None:
        self._generate_packet()

    def serialize_state_dict(self, state_dict):
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        
        compressed_buffer = BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as f:
            f.write(buffer.getvalue())
        
        compressed_data = compressed_buffer.getvalue()
        compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
        print(f"Serialized and compressed state_dict size: {len(compressed_data)} bytes")
        
        return json.dumps(compressed_base64)

    def handle_packet(self, message: str) -> None:
        simple_message: SimpleMessage = json.loads(message)

        self._log.info(simple_message)
        if simple_message['sender_type'] == SimpleSender.UAV.value and self.model_updated:
            print(f"\n\n [{self.id}] Local Model updated! \n\n")
            self.model_updated = False
            response: SimpleMessage = {
                'payload': self.serialize_state_dict(self.current_state),
                'sender': self.id,
                'packet_count': self.packet_count,
                'type': 'model_update',
                'training_cycles': self.training_cycles,
                'model_updates': self.model_updates,
                'success_rate': self.communicator.successful_attempts / self.communicator.total_attempts if self.communicator.total_attempts > 0 else 0
            }

            command = SendMessageCommand(json.dumps(response), simple_message['sender'])
            self.communicator.send_message(command, self.provider)

            self.packet_count += 1
        if simple_message['type'] == 'model_update':
            print(f"\n\n [{self.id}] Got model update from UAV! \n\n")
            state = decompress_and_deserialize_state_dict(simple_message['payload'])
            self.global_model.load_state_dict(state)
            self.global_model_changed = True
            self.model_updates += 1

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        self.finished = True
        self._log.info(f"Final packet count: {self.packet_count}")
        self.communicator.log_metrics()

def decompress_and_deserialize_state_dict(serialized_state_dict):
    compressed_data = base64.b64decode(serialized_state_dict.encode('utf-8'))
    compressed_buffer = BytesIO(compressed_data)
    with gzip.GzipFile(fileobj=compressed_buffer, mode='rb') as f:
        buffer = BytesIO(f.read())
    buffer.seek(0)
    return torch.load(buffer)
