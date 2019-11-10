import argparse
from models.inception import *

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--data_path', type=str, default='data/amateur/IMAGES', help='Path to the training set')
    parser.add_argument('--model', type=str, default='inceptionv3', help='Model to run')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing')

    args = parser.parse_args()
    return args

def main_handler(args):
    if args.model == 'inceptionv3':
        print("Using Inception V3 model")


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    main_handler(args)
    