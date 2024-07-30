from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from uav_protocol import SimpleUAVProtocol
from sensor_protocol import SimpleSensorProtocol
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--duration', type=int, default=5000, help='Duration')
    parser.add_argument('--mode', type=str, default='autoencoder', choices=['autoencoder', 'supervisioned'], help='Training mode')

    args = parser.parse_args()

    # Configuring simulation
    config = SimulationConfiguration(
        duration=args.duration,
        execution_logging=False
    )
    builder = SimulationBuilder(config)

    # Set the training mode for the protocols
    SimpleUAVProtocol.training_mode = args.mode
    SimpleSensorProtocol.training_mode = args.mode

    # Instantiating 4 sensors in fixed positions
    builder.add_node(SimpleSensorProtocol, (150, 0, 0))
    builder.add_node(SimpleSensorProtocol, (0, 150, 0))
    builder.add_node(SimpleSensorProtocol, (-150, 0, 0))
    builder.add_node(SimpleSensorProtocol, (0, -150, 0))

    # Instantiating 1 UAV at (0,0,0)
    builder.add_node(SimpleUAVProtocol, (0, 0, 0))

    # Adding required handlers
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

    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()

if __name__ == "__main__":
    main()
