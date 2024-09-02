from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from uav_protocol import SimpleUAVProtocol
from sensor_protocol import SimpleSensorProtocol
import argparse

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



# Example usage
mission_list = [(0, 0, 20), (150, 0, 20), (0, 0, 20)]  # Just an example mission list

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Federated Learning with Autoencoders on CIFAR-10')
    parser.add_argument('--duration', type=int, default=5000, help='Duration')
    parser.add_argument('--mode', type=str, default='autoencoder', choices=['autoencoder', 'supervisioned'], help='Training mode')
    parser.add_argument('--from_scratch', action='store_true', help='Start training from scratch without loading a pre-trained model')
    parser.add_argument('--resume', dest='from_scratch', action='store_false', help='Do not start from scratch, use pre-trained model')
    parser.add_argument('--success_rate', type=float, default=1.0, help='Communication success rate (0.0 to 1.0)')

    args = parser.parse_args()

    print(f"Args: {args}")

    config = SimulationConfiguration(
        duration=args.duration,
        execution_logging=False
    )
    builder = SimulationBuilder(config)

    # Passing the communication success rate to the sensors
    builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (150, 0, 0))
    builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (0, 150, 0))
    builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (-150, 0, 0))
    builder.add_node(create_protocol_with_params(SimpleSensorProtocol, training_mode = args.mode, from_scratch = args.from_scratch, success_rate = args.success_rate), (0, -150, 0))

    uav = create_protocol_with_params(SimpleUAVProtocol, training_mode = args.mode, from_scratch = args.from_scratch)
    builder.add_node(uav, (0, 0, 0))

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

# self.provider.tracked_variables['blabla'] = 'oi'