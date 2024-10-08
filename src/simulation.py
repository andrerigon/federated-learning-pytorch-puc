import argparse
import math
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from uav_protocol import SimpleUAVProtocol
from sensor_protocol import SimpleSensorProtocol
import random

def create_protocol_with_params(protocol_class, **init_params):
    """
    Dynamically creates a class that wraps the given protocol_class and injects the init_params
    """
    class ProtocolWrapper(protocol_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for key, value in init_params.items():
                setattr(self, key, value)
    return ProtocolWrapper

sensor_positions = [(150, 0, 0), (0, 150, 0), (-150, 0, 0), (0, -150, 0)]

def generate_mission_list(num_uavs: int, sensor_positions: list) -> list:
    """
    Generates a mission list for each UAV ensuring that each UAV visits all sensor positions.

    Args:
        num_uavs: Number of UAVs.
        sensor_positions: List of tuples representing the (x, y, z) positions of the sensors.

    Returns:
        List of mission lists for each UAV.
    """
    # Center position where all UAVs meet after visiting sensors
    center_position = (0, 0, 0)

    # Calculate the number of sensors per UAV
    sensors_per_uav = len(sensor_positions) // num_uavs
    extra_sensors = len(sensor_positions) % num_uavs

    mission_lists = []

    sensor_index = 0
    for uav_index in range(num_uavs):
        # Assign sensors to the UAV
        uav_mission = []
        for _ in range(sensors_per_uav):
            uav_mission.append(sensor_positions[sensor_index])
            sensor_index += 1
        
        # Distribute extra sensors
        if uav_index < extra_sensors:
            uav_mission.append(sensor_positions[sensor_index])
            sensor_index += 1

        # Add the center position to meet other UAVs
        uav_mission.append(center_position)

        # No need to add the return journey manually since `LoopMission.REVERSE` will handle it.

        mission_lists.append(uav_mission)

    return mission_lists

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Federated Learning with Autoencoders on CIFAR-10')
    parser.add_argument('--duration', type=int, default=5000, help='Duration')
    parser.add_argument('--mode', type=str, default='autoencoder', choices=['autoencoder', 'supervisioned'], help='Training mode')
    parser.add_argument('--from_scratch', action='store_true', help='Start training from scratch without loading a pre-trained model')
    parser.add_argument('--resume', dest='from_scratch', action='store_false', help='Do not start from scratch, use pre-trained model')
    parser.add_argument('--success_rate', type=float, default=1.0, help='Communication success rate (0.0 to 1.0)')
    parser.add_argument('--num_uavs', type=int, default=4, help='Number of UAVs in the simulation')

    args = parser.parse_args()

    print(f"Args: {args}")

    config = SimulationConfiguration(
        duration=args.duration,
        execution_logging=False
    )
    builder = SimulationBuilder(config)

    mission_lists = generate_mission_list(args.num_uavs, sensor_positions)

    builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (150, 0, 0))
    builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (0, 150, 0))
    builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (-150, 0, 0))
    builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (0, -150, 0))

    # Add UAVs to the simulation, starting them at random positions
    for i in range(args.num_uavs):
        start_position = mission_lists[i][0]  # Use the first position in the mission list as the start position
        uav_protocol = create_protocol_with_params(SimpleUAVProtocol, training_mode=args.mode, from_scratch=args.from_scratch, uav_id=i + 1, mission_list=mission_lists[i])
        builder.add_node(uav_protocol, start_position)
    
    # Add handlers as before
    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=30
    )))
    builder.add_handler(MobilityHandler())
    builder.add_handler(VisualizationHandler(VisualizationConfiguration(
        x_range=(-200, 200),
        y_range=(-200, 200),
        z_range=(0, 200),
        open_browser=False
    )))

    simulation = builder.build()
    simulation.start_simulation()

if __name__ == "__main__":
    main()
