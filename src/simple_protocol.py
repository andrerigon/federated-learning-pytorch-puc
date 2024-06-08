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


class ModelUpdate:
    state_dict_json: str
    sender: int


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
        try:
            local_model = Autoencoder().to(device)
            local_model.load_state_dict(self.global_model.state_dict())
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
        self._log.info(f'sensor received from sender: {simple_message['sender_type']} and mode updated is {self.model_updated}')
        if simple_message['sender_type'] == SimpleSender.UAV.value and self.model_updated:
            self.model_updated = False
            # response: SimpleMessage = {
            #     'packet_count': self.packet_count,
            #     'sender_type': SimpleSender.SENSOR.value,
            #     'sender': self.provider.get_id()
            # }
            response: ModelUpdate = {
                'state_dict_json': self.serialize_state_dict(self.current_state),
                'sender': self.id
            }



            command = SendMessageCommand(json.dumps(response), simple_message['sender'])
            self.provider.send_communication_command(command)

            # self._log.info(f"Sent {response['packet_count']} packets to UAV {simple_message['sender']}")

            self.packet_count = 0

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


class SimpleUAVProtocol(IProtocol):
    _log: logging.Logger

    packet_count: int

    _mission: MissionMobilityPlugin

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.packet_count = 0

        self._mission = MissionMobilityPlugin(self, MissionMobilityConfiguration(
            loop_mission=LoopMission.REVERSE,
        ))

        self._mission.start_mission(mission_list.pop())

        self._send_heartbeat()

    def _send_heartbeat(self) -> None:
        # self._log.info(f"Sending heartbeat, current count {self.packet_count}")

        message: SimpleMessage = {
            'packet_count': self.packet_count,
            'sender_type': SimpleSender.UAV.value,
            'sender': self.provider.get_id()
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

        self.provider.schedule_timer("", self.provider.current_time() + 1)

    def handle_timer(self, timer: str) -> None:
        self._send_heartbeat()

    def handle_packet(self, message: str) -> None:
        self._log.info(f'got message with size: {len(message)}')
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
        self._log.info(f"Final packet count: {self.packet_count}")


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
