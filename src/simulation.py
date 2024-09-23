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
from sklearn.cluster import KMeans
import numpy as np

def create_protocol_with_params(protocol_class, class_name=None, **init_params):
    """
    Dynamically creates a class that wraps the given protocol_class and injects the init_params,
    allowing you to specify the class name of the wrapper.
    """
    # If no class_name is provided, default to protocol_class's name with 'Wrapper' appended
    if class_name is None:
        class_name = protocol_class.__name__

    # Define the __init__ method for the new class
    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the parent class
        super(self.__class__, self).__init__(*args, **kwargs)
        # Set the initial parameters as attributes
        for key, value in init_params.items():
            setattr(self, key, value)

    # Create a dictionary of class attributes
    class_attrs = {'__init__': __init__}

    # Dynamically create the new class with the specified name
    new_class = type(class_name, (protocol_class,), class_attrs)

    return new_class

def cluster_sensors(sensor_positions: list, num_uavs: int):
    sensor_coords = np.array([pos[:2] for pos in sensor_positions])  # Use x, y positions
    kmeans = KMeans(n_clusters=num_uavs, random_state=0).fit(sensor_coords)
    labels = kmeans.labels_
    clusters = [[] for _ in range(num_uavs)]
    for idx, label in enumerate(labels):
        clusters[label].append(sensor_positions[idx])
    return clusters

def generate_mission_list(num_uavs: int, sensor_positions: list, repetitions: int = 1) -> list:
    """
    Generates a mission list for each UAV where each UAV repeatedly visits its assigned sensors,
    goes to the center, and restarts the loop for a specified number of repetitions.

    Args:
        num_uavs: Number of UAVs.
        sensor_positions: List of tuples representing the (x, y, z) positions of the sensors.
        repetitions: Number of times each UAV should repeat its mission loop.

    Returns:
        List of mission lists for each UAV.
    """
    # Center position where all UAVs meet after visiting sensors
    center_position = (0, 0, 0)

    # Cluster sensors based on their positions
    clusters = cluster_sensors(sensor_positions, num_uavs)

    mission_lists = []

    for uav_index in range(num_uavs):
        # Assign the cluster to the UAV
        uav_sensors = clusters[uav_index]

        # Create the repeated mission sequence
        uav_mission = []
        for _ in range(repetitions):
            uav_mission.extend(uav_sensors)
            uav_mission.append(center_position)

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
    parser.add_argument('--num_uavs', type=int, default=1, help='Number of UAVs in the simulation')

    args = parser.parse_args()

    print(f"Args: {args}")

    sensor_positions = [(150, 0, 0), (0, 150, 0), (-150, 0, 0), (0, -150, 0)]

    config = SimulationConfiguration(
        duration=args.duration,
        execution_logging=False
    )
    builder = SimulationBuilder(config)

    mission_lists = generate_mission_list(args.num_uavs, sensor_positions)

    sensor_ids: list[int] = []

    sensor_ids.append(builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (150, 0, 0)))
    sensor_ids.append(builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (0, 150, 0)))
    sensor_ids.append(builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (-150, 0, 0)))
    sensor_ids.append(builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (0, -150, 0)))

    # Add UAVs to the simulation, starting them at random positions
    leader_ids = []
    for i in range(args.num_uavs):
        start_position = mission_lists[i][0]  # Use the first position in the mission list as the start position
        uav_protocol = create_protocol_with_params(SimpleUAVProtocol, training_mode=args.mode, from_scratch=args.from_scratch, uav_id=i + 1, mission_list=mission_lists[i])
        leader_ids.append(builder.add_node(uav_protocol, start_position))
    
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

    node = simulation.get_node(sensor_ids[0])
    print(f"\n\n\nprotocol: {node.protocol_encapsulator.protocol.bla()}\n\n\n")


    positions = []
    sensor_positions = []

    for sensor_id in sensor_ids:
        sensor_position = simulation.get_node(sensor_id).position
        sensor_positions.append({
            "role": "sensor",
            "x": sensor_position[0],
            "y": sensor_position[1],
            "z": sensor_position[2],
        })

    while simulation.step_simulation():
        current_time = simulation._current_timestamp # Não gostei disso, vou tornar uma propriedade mais fácil de acessar

        for leader_id in leader_ids:
            leader_position = simulation.get_node(leader_id).position
            positions.append({
                "role": "UAV",
                "agent": leader_id,
                "timestamp": current_time,
                "x": leader_position[0],
                "y": leader_position[1],
                "z": leader_position[2],
            })

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    position_df = pd.DataFrame.from_records(positions)
    position_df = position_df.set_index("timestamp")

    sensor_df = pd.DataFrame.from_records(sensor_positions)

    # Estilizando gráfico
    sns.set_theme()
    sns.set_context("talk")
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plotando posições dos sensores (fixos) com um "x"
    sns.scatterplot(data=sensor_df, x="x", y="y", ax=ax, marker='x', color='black',
                    label='Sensors', s=100, linewidth=2)

    # Plotando as posições dos agentes ao longo do tempo. Líder em vermelho e seguidores e mazul
    grouped = position_df.groupby("agent")
    for name, group in grouped:
        role = group["role"].iloc[0]
        plt.plot(group['x'], group['y'], marker='o', linestyle='-', ms=1,
                 label=role, color='#cf7073' if role == "leader" else '#4c72b0')

    # Mostrando legenda. As duas primeiras linhas garantem que não vão ter elementos repetidos na legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Salvando o gráfico
    plt.savefig("path.png")

if __name__ == "__main__":
    main()
