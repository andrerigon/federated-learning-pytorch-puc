import json
import logging
import random

import torch.nn as nn
import os

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry

from image_classifier_autoencoders import Autoencoder, device as ae_device
from image_classifier_supervisioned import SupervisedModel, device as sup_device
import torch.optim as optim
from tqdm import tqdm
import gc
from io import BytesIO

import threading
import torch
import torch.multiprocessing as mp
import multiprocessing.shared_memory as shm
from messages import SimpleSender, SimpleMessage
from communication import CommunicationMediator
from serialization import decompress_and_deserialize_state_dict, serialize_state_dict

def report_message(message: SimpleMessage) -> str:
    return ''

class SimpleSensorProtocol(IProtocol):
    _log: logging.Logger
    packet_count: int
    remaining_energy: int

    def __init__(self, dataset_loader = None, global_model = None, training_mode = "autoencoder", success_rate = 1.0):
        self.training_mode = training_mode
        self.success_rate = success_rate
        self.dataset_loader = dataset_loader
        self.global_model = global_model

    def initialize(self) -> None:
        self.remaining_energy = random.randint(1, 5)
        self._log = logging.getLogger()
        self.packet_count = 0
        
        self.id = self.provider.get_id()
        self.global_model_changed = False

        self.loader = self.dataset_loader.loader(client_id= self.id)
        self.testset = self.dataset_loader.testset

        self.training_cycles = 0
        self.model_updates = 0
        self.global_model_version = 0

        # Communication Mediator with configurable success rate
        self.communicator = CommunicationMediator[SendMessageCommand](success_rate = self.success_rate)

        self.thread = threading.Thread(target=self.start_training)
        self.finished = False
        self.model_updated = False
        self.thread.start()

    
    def start_training(self):
        while not self.finished:
            self.train_autoencoder()

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
                progress_bar = tqdm(enumerate(self.loader, 0), total=len(self.loader), desc=f'Sensor {self.id+1}, Cycle {self.training_cycles+1}', leave=False)
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
                    # Release batch memory
                    del inputs, decoded, classified, loss, labels
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
            
            del local_model, criterion_reconstruction, optimizer, scheduler
            # gc.collect()

        except Exception as e:
            logging.error(f"Error in client {self.id}", exc_info=True)

    def bla(self):
        print(f"\n\nme meg todim todim\n")

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
        pass
        # self._generate_packet()


    def handle_packet(self, message: str) -> None:
        simple_message: SimpleMessage = json.loads(message)

        self._log.info(simple_message)
        
        # If the message is from a UAV
        if simple_message['sender_type'] == SimpleSender.UAV.value:
            # If it's a model update message
            if simple_message['type'] == 'model_update':
                new_version = simple_message.get('global_model_version', 0)
                if new_version <= self.global_model_version:
                    # Discard the message
                    print(f"Sensor {self.id}: Discarding model update from UAV {simple_message['sender']} with version {new_version} as it's not newer.")
                    return  # Exit without processing
                # Else, process the model update
                print(f"Sensor {self.id}: Received new model update from UAV {simple_message['sender']} with version {new_version}")
                state = decompress_and_deserialize_state_dict(simple_message['payload'])
                self.global_model.load_state_dict(state)
                self.global_model_changed = True
                self.model_updates += 1
                self.global_model_version = new_version
                # # Set a flag to start training with the new model
                # self.model_updated = True
                del state

            # If it's a ping and the local model is updated
            elif simple_message['type'] == 'ping' and self.model_updated:
                new_version = simple_message.get('global_model_version', 0)
                if new_version <= self.global_model_version:
                    # Discard the message
                    self._log.info(f"Sensor {self.id}: Discarding model update from UAV {simple_message['sender']} with version {new_version} as it's not newer.")
                    return  # Exit without processing
                self.model_updated = False
                self._log.info(f"Sensor {self.id}: Responding to ping from UAV {simple_message['sender']}")
                response: SimpleMessage = {
                    'payload': serialize_state_dict(self.current_state),
                    'sender': self.id,
                    'sender_type': SimpleSender.SENSOR.value,
                    'packet_count': self.packet_count,
                    'type': 'model_update',
                    'training_cycles': self.training_cycles,
                    'model_updates': self.model_updates,
                    'success_rate': self.communicator.successful_attempts / self.communicator.total_attempts if self.communicator.total_attempts > 0 else 0,
                    'local_model_version': self.global_model_version  # Include version used for training
                }
                command = SendMessageCommand(json.dumps(response), simple_message['sender'])
                self.communicator.send_message(command, self.provider)
                self.packet_count += 1
                del command, response
        del simple_message, message
        # gc.collect() 

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        """
        Clean up resources and terminate any active threads.
        """
        self.finished = True
        self._log.info(f"Final packet count: {self.packet_count}")
        self.communicator.log_metrics()

        # Wait for the training thread to complete
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        
        # Ensure all torch resources are freed
        torch.cuda.empty_cache()  # Clear CUDA memory if using GPU
        self.global_model.cpu()   # Move model to CPU to free GPU memory
        del self.global_model     # Remove model to free memory

        # Logging final status
        self.loader._iterator = None
        del self.loader
        # gc.collect()
        self._log.info("Sensor Protocol finished and cleaned up.")

