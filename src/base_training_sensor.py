from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry

class BaseTrainingSensor(IProtocol):
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
            self.run_training()

    def run_training(self):
        raise 'Not implementd'

