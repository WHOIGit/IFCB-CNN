# Keras_IFCBCNN


The code was tested on Python 3.5, with the following library versions,
Keras 2.0.6
TensorFlow 1.2.1s
OpenCV 3.2.0

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created [label]/ subfolders inside train/ and validation/
So that we have 4000 training examples for each class, and 1000 validation examples for each class.
In summary, this is our directory structure:

```
data/
    train/
        plankton1/
            IFCB001.jpg
            IFCB002.jpg
            ...
        plankton2/
            IFCB001.jpg
            IFCB002.jpg
            ...
    valid/
        plankton1/
            IFCB001.jpg
            IFCB002.jpg
            ...
        plankton2/
            IFCB001.jpg
            IFCB002.jpg
            ...
    test/
        plankton1/
            IFCB001.jpg
            IFCB002.jpg
            ...
        plankton2/
            IFCB001.jpg
            IFCB002.jpg
            ...
```

### Training

The Script `transfer-train.py` uses hardcoded directories and is included for archival purposes. To run on your own
dataset, use `simple-transfer-train.py`:
```
python simple-transfer-train.py --train_dir=/data/plank_10_fixed/train --val_dir=/data/plank_10_fixed/valid --output_model_file="inception_10class.model" --nb_epoch=25 --plot
```
This will run transfer-learning using inceptionv3 weights downloaded automatically, using the defined train and validation directories,
defined output file, 25 epochs (cycles of training and validation), and plots the progress.
For full training of the network use `train.py` with the same arguments but set `--freeze_layers_number=1`

### Predictions 

Single image predictions with `predict.py`:
```
python predict.py --path "/full/path/to/image" --model=inception
```

Generate predictions with test folder from dataset and generate confusion matrix with `predict_cnfmatrix.py` using a saved model:
```
python predict.py --test_dir "/full/path/to/test/folder" --model="path/to/model.model
```

