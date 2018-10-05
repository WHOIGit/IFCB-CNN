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
python train.py --model=resnet50
```

For full training of the network use `train.py`

### Predictions 

Single image predictions with `predict.py`:
```
python predict.py --path "/full/path/to/image" --model=inception
```

Generate predictions with test folder from dataset and generate confusion matrix with `predict_cnfmatrix.py`:
```
python predict.py --path "/full/path/to/image" --model=resnet50
```

