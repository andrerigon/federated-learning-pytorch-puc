from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from simple_protocol import SimpleSensorProtocol, SimpleGroundStationProtocol, SimpleUAVProtocol



def main():
    # Configuring simulation
    config = SimulationConfiguration(
        duration=10000
    )
    builder = SimulationBuilder(config)

    # Instantiating 4 sensors in fixed positions
    builder.add_node(SimpleSensorProtocol, (150, 0, 0))
    builder.add_node(SimpleSensorProtocol, (0, 150, 0))
    builder.add_node(SimpleSensorProtocol, (-150, 0, 0))
    builder.add_node(SimpleSensorProtocol, (0, -150, 0))

    # Instantiating 4 UAVs at (0,0,0)
    builder.add_node(SimpleUAVProtocol, (0, 0, 0))
    # builder.add_node(SimpleUAVProtocol, (0, 0, 0))
    # builder.add_node(SimpleUAVProtocol, (0, 0, 0))
    # builder.add_node(SimpleUAVProtocol, (0, 0, 0))

    # # Instantiating ground station at (0,0,0)
    # builder.add_node(SimpleGroundStationProtocol, (0, 0, 0))

    # Adding required handlers
    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=30
    )))
    builder.add_handler(MobilityHandler())
    builder.add_handler(VisualizationHandler(VisualizationConfiguration(
        x_range=(-150, 150),
        y_range=(-150, 150),
        z_range=(0, 150)
    )))

    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()

import torch
import base64
import json
from io import BytesIO
from collections import OrderedDict

def serialize_state_dict(state_dict):
    serialized_dict = {}
    for key, tensor in state_dict.items():
        buffer = BytesIO()
        torch.save(tensor, buffer)
        tensor_bytes = buffer.getvalue()
        tensor_base64 = base64.b64encode(tensor_bytes).decode('utf-8')
        serialized_dict[key] = tensor_base64
    return json.dumps(serialized_dict)

def deserialize_state_dict(serialized_state_dict):
    state_dict = json.loads(serialized_state_dict)
    deserialized_dict = OrderedDict()
    for key, tensor_base64 in state_dict.items():
        tensor_bytes = base64.b64decode(tensor_base64.encode('utf-8'))
        buffer = BytesIO(tensor_bytes)
        tensor = torch.load(buffer)
        deserialized_dict[key] = tensor
    return deserialized_dict

if __name__ == "__main__":
    main()