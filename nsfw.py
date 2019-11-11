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

    args = parser.parse_args()
    return args

def main_handler(args):
    if args.model == 'inceptionv3':
        print("Using Inception V3 model")
        dataset = NSFWDataset(args.nsfw_path, args.neutral_data_path)
        
        fig = plt.figure()

        for i in range(len(dataset)):
            sample = dataset[i]

            #print(i, sample[0].shape)
            print(sample[1])

            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            ax.imshow(sample[0])

            if i == 3:
                plt.show()
                break


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    main_handler(args)
    