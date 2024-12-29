import json
import random
from loguru import logger

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry

from messages import SimpleSender, SimpleMessage
from communication import CommunicationMediator
from serialization import decompress_and_deserialize_state_dict, serialize_state_dict
from federated_learning_trainer import FederatedLearningTrainer

def report_message(message: SimpleMessage) -> str:
    return ''

class SimpleSensorProtocol(IProtocol):
    packet_count: int
    remaining_energy: int

    def __init__(self, federated_learning_trainer: FederatedLearningTrainer = None,  success_rate = 1.0):
        self.federated_learning_trainer = federated_learning_trainer
        self.success_rate = success_rate

    def initialize(self) -> None:
        self.remaining_energy = random.randint(1, 5)
        self.packet_count = 0
        
        self.id = self.provider.get_id()
        self.logger = logger.bind(source="sensor", sensor_id=self.id)

        # Communication Mediator with configurable success rate
        self.communicator = CommunicationMediator[SendMessageCommand](success_rate = self.success_rate)
        self.logger.info("Started")
    
    def bla(self):
        print(f"\n\nme meg todim todim\n")

    def handle_timer(self, timer: str) -> None:
        self._generate_packet()

    def handle_packet(self, message: str) -> None:
        simple_message: SimpleMessage = json.loads(message)
        
        # If the message is from a UAV
        if simple_message['sender_type'] == SimpleSender.UAV.value:
            # If it's a model update message
            if simple_message['type'] == 'model_update':
                new_version = simple_message.get('global_model_version', -1)
                if new_version <= self.federated_learning_trainer.global_model_version:
                    # Discard the message
                    self.logger.info(f"Discarding model update from UAV {simple_message['sender']} with version {new_version} as it's not newer.")
                    return  # Exit without processing
                
                # Else, process the model update
                self.logger.info(f"Received new model update from UAV {simple_message['sender']} with version {new_version}")
                state = decompress_and_deserialize_state_dict(simple_message['payload'])
                self.federated_learning_trainer.update_model(state, new_version)
                del state

            # If it's a ping and the local model is updated
            elif simple_message['type'] == 'ping':
                new_version = simple_message.get('global_model_version', 0)
                # if new_version <= self.federated_learning_trainer.last_version():
                #     # Discard the message
                #     self.logger.debug(f"Discarding model update from UAV {simple_message['sender']} with version {new_version} as it's not newer.")
                #     return  # Exit without processing
                if self.federated_learning_trainer.current_state == None or not self.federated_learning_trainer.model_updated: 
                    self.logger.debug("No local model is available")
                    return 
                self.logger.info(f"Responding to ping from UAV {simple_message['sender']}")
                response: SimpleMessage = {
                    'payload': serialize_state_dict(self.federated_learning_trainer.current_state),
                    'sender': self.id,
                    'sender_type': SimpleSender.SENSOR.value,
                    'packet_count': self.packet_count,
                    'type': 'model_update',
                    'training_cycles': self.federated_learning_trainer.training_cycles,
                    'model_updates': self.federated_learning_trainer.model_updates,
                    'success_rate': self.communicator.successful_attempts / self.communicator.total_attempts if self.communicator.total_attempts > 0 else 0,
                    'local_model_version': self.federated_learning_trainer.last_version()  # Include version used for training
                }
                command = SendMessageCommand(json.dumps(response), simple_message['sender'])
                self.communicator.send_message(command, self.provider)
                self.packet_count += 1
                self.federated_learning_trainer.model_updated = False
                del command, response
        del simple_message, message
        # gc.collect() 

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        """
        Clean up resources and terminate any active threads.
        """
        self.federated_learning_trainer.stop()
        self.logger.info(f"Final packet count: {self.packet_count}")
        self.communicator.log_metrics()

        self.logger.info("Sensor Protocol finished and cleaned up.")

