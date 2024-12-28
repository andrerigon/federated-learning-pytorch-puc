import json
import logging
import queue
import threading
from datetime import datetime
from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.plugin.mission_mobility import MissionMobilityPlugin, MissionMobilityConfiguration, LoopMission
from typing import List, Dict
from sensor_protocol import SimpleMessage, SimpleSender
from federated_learning_aggregator import FederatedLearningAggregator
from serialization import decompress_and_deserialize_state_dict, serialize_state_dict
from loguru import logger

######################################################
# Convergence Criteria
######################################################

class ConvergenceCriteria:
    """
    Base class for convergence checking.
    """
    def has_converged(self, metrics_history: List[Dict[str, float]]) -> bool:
        raise NotImplementedError("Implement in subclass")

class AccuracyConvergence(ConvergenceCriteria):
    """
    Converge if accuracy surpasses threshold or no improvement in last `patience` steps.
    """
    def __init__(self, threshold: float = 0.95, patience: int = 5):
        self.threshold = threshold
        self.patience = patience

    def has_converged(self, metrics_history: List[Dict[str, float]]) -> bool:
        accuracy_values = [m['accuracy'] for m in metrics_history if 'accuracy' in m]
        logger.info(f"Last accuracy: {accuracy_values[-1]} threshold: {self.threshold} result: {accuracy_values[-1] >= self.threshold} last values: {accuracy_values}\n")
        if not accuracy_values:
            return False
        if accuracy_values[-1] >= self.threshold:
            return True
        # if len(accuracy_values) > self.patience:
        #     recent = accuracy_values[-self.patience:]
        #     if all(a <= accuracy_values[-self.patience - 1] for a in recent):
        #         return True
        return False

class StabilityConvergence(ConvergenceCriteria):
    """
    Converge if loss stabilizes within `tol` for last `window` measurements.
    """
    def __init__(self, tol: float = 1e-3, window: int = 5):
        self.tol = tol
        self.window = window

    def has_converged(self, metrics_history: List[Dict[str, float]]) -> bool:
        loss_values = [m['loss'] for m in metrics_history if 'loss' in m]
        if len(loss_values) < self.window:
            return False
        recent = loss_values[-self.window:]
        return max(recent) - min(recent) < self.tol


class SimpleUAVProtocol(IProtocol):
    """
    UAV Protocol that:
    - Relies on FederatedLearningAggregator for model aggregation and evaluation.
    - Receives model updates from sensors (clients) and passes them to the aggregator.
    - Uses time-based FedAvg or other strategies via the aggregator.
    - The UAV no longer directly manages model logic or strategies.
    """

    def __init__(self, aggregator:FederatedLearningAggregator  = None, mission_list=[(0, 0, 20),(150, 0, 20),(0, 0, 20),(0, 150, 20),(0, 0, 20),(-150, 0, 20),(0, 0, 20),(0, -150, 20)]):
        self.aggregator = aggregator
        self.mission_list = mission_list
        self.last_received_time = {}
        self.debounce_interval = 5
        self.message_queue = queue.Queue()
        self.message_processing_thread = threading.Thread(target=self.process_messages)
        self.message_processing_thread.daemon = True

    def initialize(self) -> None:
        self.packet_count = 0
        self.id = self.provider.get_id()

        self.logger = logger.bind(source="uav", uav_id=self.id)
        self.logger.info(f"Starting UAV")

        mission_config = MissionMobilityConfiguration(loop_mission=LoopMission.RESTART)
        self._mission = MissionMobilityPlugin(self, mission_config)
        self._mission.start_mission(self.mission_list)

        self._send_heartbeat()

        # Start aggregator evaluation thread
        self.aggregator.start()

        self.message_processing_thread.start()

    def _send_heartbeat(self) -> None:
        message: SimpleMessage = {
            'packet_count': self.packet_count,
            'sender_type': SimpleSender.UAV.value,
            'sender': self.provider.get_id(),
            'type': 'ping',
            'global_model_version': self.aggregator.global_model_version 
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)
        self.provider.schedule_timer("", self.provider.current_time() + 1)

    def handle_timer(self, timer: str) -> None:
        self._send_heartbeat()

    def process_messages(self):
        while True:
            message = self.message_queue.get()
            if message is None:
                break
            self.process_single_message(message)
            self.message_queue.task_done()

    def process_single_message(self, body: str):
        try:
            message: SimpleMessage = json.loads(body)
            sender_id = message['sender']

            if message['type'] == 'model_update':
                sender_type = ("Sensor" if message['sender_type'] == SimpleSender.SENSOR.value else "UAV")
                self.logger.info(f"Received model update from {sender_type} {sender_id}")

                decompressed_state_dict = decompress_and_deserialize_state_dict(message['payload'])

                client_model_version = message.get('local_model_version', 0)
                # Pass to aggregator
                self.aggregator.receive_model_update(sender_id, decompressed_state_dict, client_model_version)

                # Send global model back to that sender
                response_message: SimpleMessage = {
                    'sender_type': SimpleSender.UAV.value,
                    'payload': serialize_state_dict(self.aggregator.state_dict()),
                    'sender': self.id,
                    'packet_count': self.packet_count,
                    'type': 'model_update',
                    'global_model_version': self.aggregator.global_model_version
                }
                command = SendMessageCommand(json.dumps(response_message), sender_id)
                self.provider.send_communication_command(command)

            elif message['type'] == 'ping':
                self.logger.info(f"Received ping from {sender_id}")
                response_message: SimpleMessage = {
                    'sender_type': SimpleSender.UAV.value,
                    'payload': self.aggregator.model_manager.serialize_state_dict(self.aggregator.state_dict()),
                    'sender': self.id,
                    'packet_count': self.packet_count,
                    'type': 'model_update',
                    'global_model_version': self.aggregator.global_model_version
                }
                command = BroadcastMessageCommand(json.dumps(response_message))
                self.provider.send_communication_command(command)
            else:
                self.logger.warning(f"Unknown message type from sender {sender_id}")

        except KeyError as e:
            self.logger.error(f"KeyError: Missing key in message - {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
        # gc.collect()

    def handle_packet(self, message: str) -> None:
        self.message_queue.put(message)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        # Signal aggregator to stop and finalize
        self.aggregator.stop()
        self.logger.info(f"Finished.")