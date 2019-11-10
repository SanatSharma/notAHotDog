import argparse
from models.inception import *

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--data_path', type=str, default='data', help='Path to the training set')
    parser.add_argument('--model', type=str, default='inceptionv3', help='Model to run')

    args = parser.parse_args()
    return args

def main_handler(args):
    if args.model == 'inceptionv3':
        print("Using Inception V3 model")


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    main_handler(args)
    