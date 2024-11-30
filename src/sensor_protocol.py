import json
import logging
import random

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
    _log: logging.Logger
    packet_count: int
    remaining_energy: int

    def __init__(self, federated_learning_trainer: FederatedLearningTrainer = None,  success_rate = 1.0):
        self.federated_learning_trainer = federated_learning_trainer
        self.success_rate = success_rate

    def initialize(self) -> None:
        self.remaining_energy = random.randint(1, 5)
        self._log = logging.getLogger()
        self.packet_count = 0
        
        self.id = self.provider.get_id()

        # Communication Mediator with configurable success rate
        self.communicator = CommunicationMediator[SendMessageCommand](success_rate = self.success_rate)
    
    def bla(self):
        print(f"\n\nme meg todim todim\n")

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
                if new_version <= self.federated_learning_trainer.global_model_version:
                    # Discard the message
                    print(f"Sensor {self.id}: Discarding model update from UAV {simple_message['sender']} with version {new_version} as it's not newer.")
                    return  # Exit without processing
                # Else, process the model update
                print(f"Sensor {self.id}: Received new model update from UAV {simple_message['sender']} with version {new_version}")
                state = decompress_and_deserialize_state_dict(simple_message['payload'])
                self.federated_learning_trainer.update_model(state, new_version)
                del state

            # If it's a ping and the local model is updated
            elif simple_message['type'] == 'ping' and self.federated_learning_trainer.model_updated:
                new_version = simple_message.get('global_model_version', 0)
                if new_version <= self.federated_learning_trainer.last_version():
                    # Discard the message
                    self._log.info(f"Sensor {self.id}: Discarding model update from UAV {simple_message['sender']} with version {new_version} as it's not newer.")
                    return  # Exit without processing
                self._log.info(f"Sensor {self.id}: Responding to ping from UAV {simple_message['sender']}")
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
        self._log.info(f"Final packet count: {self.packet_count}")
        self.communicator.log_metrics()

        self._log.info("Sensor Protocol finished and cleaned up.")

