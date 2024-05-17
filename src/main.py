import logging
import sys
import multiprocessing as mp
import argparse
from data import DataLoaderWrapper
from federated import FederatedLearning
from utils import setup_logging, handle_exception, capture_warnings

def main():
    # Set up logging
    setup_logging()
    sys.excepthook = handle_exception
    capture_warnings()

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Federated Learning with PyTorch on CIFAR-10')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of workers')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')

    args = parser.parse_args()

    try:
        trainset, testset = DataLoaderWrapper.download_dataset()
        federated_learning = FederatedLearning(args.num_clients, args.epochs, max_workers=args.max_workers, batch_size=args.batch_size)
        federated_learning.train_federated(trainset)
        
        testloader = DataLoaderWrapper.split_dataset(testset, 1)[0]
        federated_learning.check(testloader)
    except Exception as e:
        logging.error("Error in main function", exc_info=True)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
