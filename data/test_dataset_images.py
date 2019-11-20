import argparse
from PIL import Image
import os

def arg_parse():
    parser = argparse.ArgumentParser(description='test_dataset.py')
    parser.add_argument('--path', nargs='*', default=['data/neutral/IMAGES'], help='Paths to image directories')
    args = parser.parse_args()
    return args


def check_all_images(file_path):
    files = [os.path.join(file_path, p) for p in sorted(os.listdir(file_path))]
    print(file_path, len(files))
    print("Opening all files")
    for file in files:
        try:
            img = Image.open(file)
        except:
            print(file)

if __name__ == '__main__':
    args = arg_parse()
    print(args)
    for path in args.path:
        check_all_images(path)