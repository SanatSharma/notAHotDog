import argparse
from models.inception import *
from data.dataset import *
import matplotlib.pyplot as plt

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--nsfw_path', nargs='*', default=['data/amateur/IMAGES'], help='Paths to the nsfw images')
    parser.add_argument('--neutral_data_path', type=str, default='data/neutral/IMAGES', help="Path to neutral images")
    parser.add_argument('--model', type=str, default='inceptionv3', help='Model to run')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--model_path', type=str, default="model/trained.pt", help='Path to save trained model')

    args = parser.parse_args()
    return args

def main_handler(args):
    if args.model == 'inceptionv3':
        print("Using Inception V3 model")
        train_data, test_data = create_nsfw_dataset(args.nsfw_path, args.neutral_data_path, args)
        print(len(train_data), len(test_data))

        print("Training")
        trained_model = train_network(train_data, args)
        print("Evaluating")
        trained_model.evaluate(test_data)
        model = trained_model.model
        
        torch.save(model.state_dict(), args.model_path)

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())

if __name__ == '__main__':
    args = arg_parse()
    print(args)
    main_handler(args)
    