# notAHotDog

Remember Hot-dog or Not Hot-dog from Silicon Valley. Well this is a Not Safe For Work classifier that finds and replaces all NSFW content with cat pics.

To scrapte data, run ```scrape_data.sh```

### If you are using Windows, remove carriage return character first, before running scraping script. It will be painful otherwise
```
dos2unix file_name.txt
```
or
```
tr -d '\r' < filewithcarriagereturns > filewithoutcarriagereturns
```

To install all required packages, run
```
pip install -r requirements.txt
```

Main file is ```nsfw.py```

We utilize InceptionV3 model trained on Imagenet as our base model for classification.
Parameters for training and testing neural architecture.
```
  '--nsfw_path', nargs='*', default=['data/amateur/IMAGES'], help='Paths to the nsfw images'
  '--neutral_data_path', type=str, default='data/neutral/IMAGES', help="Path to neutral images"
  '--model', type=str, default='inceptionv3', help='Model to run'
  '--epochs', type=int, default=5, help="training epochs"
  '--lr', type=float, default=0.001, help="learning rate for optimizer"
  '--batch_size', type=int, default=32, help='Batch size for training'
  '--test_batch_size', type=int, default=8, help='Batch size for testing'
```
