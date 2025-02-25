import argparse
import math
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from aggregation_strategy import AlphaWeightedStrategy
from uav_protocol import SimpleUAVProtocol, AccuracyConvergence
from sensor_protocol import SimpleSensorProtocol
from metrics_logger import MetricsLogger
import random
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import datetime
import gc
import subprocess
from loguru import logger
import sys
from tqdm import tqdm

from dataset_loader import DatasetLoader
from model_manager import ModelManager
from image_classifier_autoencoders import  Autoencoder
from federated_learning_trainer import FederatedLearningTrainer
from federated_learning_aggregator import FederatedLearningAggregator
from aggregation_strategy import FedAvgStrategy, AsyncFedAvgStrategy, RELAYStrategy, SAFAStrategy, AstraeaStrategy, TimeWeightedStrategy, FedProxStrategy


def custom_format(record):
    """
    Build the log line manually, adding [uav_id=..., sensor_id=...]
    only if they exist in 'record["extra"]', and apply Loguru color tags.
    """
    # Basic fields
    t          = record["time"].strftime("%H:%M:%S")
    level_name = record["level"].name
    name       = record["name"]
    function   = record["function"]
    line       = record["line"]
    message    = record["message"]

    # Check for extra fields
    extra = record["extra"]
    bracket_parts = []
    if "uav_id" in extra:
        bracket_parts.append(f"U[{extra['uav_id']}]")
    if "sensor_id" in extra:
        bracket_parts.append(f"S[{extra['sensor_id']}]")

    # Only build bracket string if we have at least one ID
    bracket_str = ""
    if bracket_parts:
        # Example: color the bracket part in <blue>
        bracket_str = f" <yellow>{', '.join(bracket_parts)}</yellow>"

    # Construct final log line using Loguru color tags
    log_line = (
        f"<green>{t}</green> <level>{level_name}</level> "
        f"<cyan>{name}</cyan>:<magenta>{line}</magenta>{bracket_str} "
        f"<level>{message}</level>\n"
    )
    return log_line

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

def distribute_sensors(num_sensors, x_range, y_range):
    """Distribute sensors randomly within the specified x and y range."""
    sensor_positions = [(random.uniform(*x_range), random.uniform(*y_range), 0) for _ in range(num_sensors)]
    return sensor_positions

def plot_path(positions, sensor_positions, output_dir):
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

    colors = {
        1:  '#1f77b4' ,
        2:  '#ff7f0e',
        3:  '#2ca02c',
        4:  '#d62728'
    }
    
    # Use 'tab20' colormap for up to 10 distinct colors
    color_palette = plt.get_cmap("tab10")
    leader_color_map = {}
    leader_index = 0
    follower_color = '#4c72b0'  # Keep the color for followers consistent

    # Assign colors to each leader dynamically
    leader_color_map = {}
    leader_index = 0

    # Plot with unique colors for each leader
    for name, group in grouped:
        role = group["role"].iloc[0]
        # Assign a color if this leader is not already in the color map
        if name not in leader_color_map:
            leader_color_map[name] = color_palette(leader_index % 20)  # Cycle if there are more than 20 leaders
            leader_index += 1
        color = leader_color_map[name]  # Use the assigned color for this leader

        plt.plot(group['x'], group['y'], marker='o', linestyle='-', ms=1, label=f"{role} {name}", color=color)


    # Mostrando legenda. As duas primeiras linhas garantem que não vão ter elementos repetidos na legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    path = os.path.join(output_dir, 'path.png')
    # Salvando o gráfico
    plt.savefig(path)


   

@logger.catch
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Federated Learning with Autoencoders on CIFAR-10')
    parser.add_argument('--training_cycles', type=int, default=10, help='Training cycles')
    parser.add_argument('--mode', type=str, default='autoencoder', choices=['autoencoder', 'supervisioned'], help='Training mode')
    parser.add_argument('--from_scratch', action='store_true', help='Start training from scratch without loading a pre-trained model')
    parser.add_argument('--resume', dest='from_scratch', action='store_false', help='Do not start from scratch, use pre-trained model')
    parser.add_argument('--success_rate', type=float, default=1.0, help='Communication success rate (0.0 to 1.0)')
    parser.add_argument('--num_uavs', type=int, default=1, help='Number of UAVs in the simulation')
    parser.add_argument('--num_sensors', type=int, default=4, help='Number of sensors to deploy')
    parser.add_argument('--target_accuracy', type=float, default=50, help='Target accuracy')
    parser.add_argument('--transmision_range', type=float, default=5, help='Device transmission range')
    parser.add_argument('--grid_size', type=float, default=200, help='Gride side size')
    parser.add_argument('--tensor_dir', type=str, default="runs", help='Tensor output dir')
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["FedAvgStrategy","AsyncFedAvgStrategy","RELAYStrategy","SAFAStrategy","AstraeaStrategy","TimeWeightedStrategy", "FedProxStrategy"],
        required=True,
        help="Name of the strategy to use (e.g., FedAvgStrategy, SAFAStrategy).",
    )

    args = parser.parse_args()

    tensor_dir = args.tensor_dir

    # Remove default sink
    logger.remove()

    # Add console sink
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=custom_format,
        level="INFO",
        colorize=True
    )

    logger.add(
        f"{tensor_dir}/sensors.log",
        level="INFO",
        format=(
        "<green>{time:HH:mm:ss}</green> "
        "[<yellow>{extra[sensor_id]}</yellow>] {message}"
        ),
        colorize=True,
        filter=lambda record: record["extra"].get("source") == "sensor"
    )

    logger.add(
        f"{tensor_dir}/uavs.log",
        level="INFO",
        format=(
        "<green>{time:HH:mm:ss}</green> "
        "[<yellow>{extra[uav_id]}</yellow>] {message}"
        ),
        colorize=True,
        filter=lambda record: record["extra"].get("source") == "uav"
    )

    logger.info(f"Args: {args}")

    # Parse the strategy name from command-line arguments
    strategy_name = args.strategy

    # Mapping strategy names to their classes
    STRATEGY_MAP = {
        "FedAvgStrategy": FedAvgStrategy(),
        "AsyncFedAvgStrategy": AsyncFedAvgStrategy(),
        "RELAYStrategy": RELAYStrategy(),
        "SAFAStrategy": SAFAStrategy(total_clients=args.num_sensors),
        "AstraeaStrategy": AstraeaStrategy(),
        "TimeWeightedStrategy": TimeWeightedStrategy(),
        "FedProxStrategy": FedProxStrategy(max(1, args.num_sensors // 10))
    } 

    # Get the strategy
    strategy = STRATEGY_MAP[strategy_name]
    
    # Check if training is synchronous
    is_synchronous = strategy_name == "FedAvgStrategy"

    # calculate 
    use_proximal_term = strategy_name == "FedProxStrategy"

    # Distribute sensors randomly in the area within x and y range (-200, 200)
    grid_size = args.grid_size
    x_range = (-grid_size, grid_size)
    y_range = (-grid_size, grid_size)
    sensor_positions = distribute_sensors(args.num_sensors, x_range, y_range)

    config = SimulationConfiguration(
        execution_logging=False
    )
    builder = SimulationBuilder(config)

    mission_lists = generate_mission_list(args.num_uavs, sensor_positions)

    dataset_loader = DatasetLoader(args.num_sensors)
    model_manager = ModelManager(
            Autoencoder, 
            # from_scratch = args.from_scratch,
            base_dir='output/autoencoder',
            num_classes=10
        )
    

    # Initialize sensor nodes
    sensor_ids = []
    sensor_id = 0
    aggregators = []
    trainers = []
    for pos in sensor_positions:
        output_dir = os.path.join(tensor_dir, f"client_{sensor_id}")
        os.makedirs(output_dir, exist_ok=True)
        metrics = MetricsLogger(client_id=sensor_id, output_dir=output_dir)    

        federated_trainer = FederatedLearningTrainer(
            sensor_id,
            model_manager,
            dataset_loader,
            metrics,
            synchronous=is_synchronous,
            use_proximal_term=use_proximal_term
        )

        trainers.append(federated_trainer)
        
        sensor_protocol = create_protocol_with_params(SimpleSensorProtocol, 
         federated_learning_trainer=federated_trainer,
         success_rate=args.success_rate)
        sensor_id = builder.add_node(sensor_protocol, pos)
        sensor_ids.append(sensor_id)
        sensor_id += 1

    # Add UAVs to the simulation
    leader_ids = []
    output_dir = os.path.join('output', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), args.mode)
    os.makedirs(output_dir, exist_ok=True)

    convergence_criteria = AccuracyConvergence(threshold=args.target_accuracy, patience=5)

    for i in range(args.num_uavs):
        uav_output_dir = os.path.join(tensor_dir, f"client_{sensor_id}")
        os.makedirs(uav_output_dir, exist_ok=True)
        metrics = MetricsLogger(client_id=sensor_id, output_dir=uav_output_dir)    

        aggregator = FederatedLearningAggregator(
            sensor_id,
            model_manager,
            dataset_loader,
            metrics,
            strategy=strategy,
            convergence_criteria=convergence_criteria,
            output_dir=tensor_dir,
            round_interval=30.0,
            client_count=args.num_sensors
        )
        aggregators.append(aggregator)

        start_position = mission_lists[i][0]  # Use the first position in the mission list as the start position
        uav_protocol = create_protocol_with_params(SimpleUAVProtocol, aggregator=aggregator , mission_list=mission_lists[i], output_dir=uav_output_dir, success_rate=args.success_rate, grid_size=grid_size)
        leader_ids.append(builder.add_node(uav_protocol, start_position))
    
    # Add handlers as before
    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=args.transmision_range)))
    builder.add_handler(MobilityHandler())
    builder.add_handler(VisualizationHandler(VisualizationConfiguration(
        x_range=x_range,
        y_range=y_range,
        z_range=(0, 200),
        open_browser=False
    )))

    simulation = builder.build()

    node = simulation.get_node(sensor_ids[0])
    print(f"\n\n\nprotocol: {node.protocol_encapsulator.protocol.bla()}\n\n\n")

    positions = []
    sensor_positions_log = []

    for sensor_id in sensor_ids:
        sensor_position = simulation.get_node(sensor_id).position
        sensor_positions_log.append({
            "role": "sensor",
            "x": sensor_position[0],
            "y": sensor_position[1],
            "z": sensor_position[2],
        })

    keep_going = True
    # training_cycles = args.training_cycles
    while simulation.step_simulation() and keep_going:
        if all(t.converged for t in aggregators):
            keep_going = False
            for a in aggregators:
                a.stop()
            for t in trainers: 
                t.stop()
            simulation._finalize_simulation()
            break
  
        current_time = simulation._current_timestamp
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

    plot_path(positions, sensor_positions_log, output_dir)
    del simulation
    gc.collect()    

if __name__ == "__main__":
    try:
        main()
    finally:
        subprocess.run("pkill -f torch_shm_manager", shell=True)