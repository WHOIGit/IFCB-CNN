# Requirements
- python 3.5.2
- Tensorflow 1.3.0

## Usage

### Prepare the image data sets
In order to start the transfer learning process, a folder named ``dataset`` needs to be created in the root of the project folder. This folder will contain the image data for the 3 categories:
* Prefeed (frames of video before feeding event)
* Duringfeed (frames of video during feeding event)
* postfeed (frames of video afte feeding event)

Create the ``dataset`` folder and add the images for all the categories in the following manner:

```
|
---- /dataset
|    |
|    |
|    ---- /prefeed
|    |    prefeed_date1.jpg
|    |    prefeed_date2.jpg
|    |    ...
|    |
|    |
|    ---- /duringfeed
|    |     duringfeed_date1.jpg
|    |     duringfeed2.jpg
|    |     ...
|    |
|    |
|    ----/postfeed
|         postfeed_date1.jpg
|         postfeed_date2.jpg
|        ...
```

For creating the datasets from GoPro videos, the script "makevids.sh" can be run from the top folder containing all the videos.
Subsequently, still image frames can be extracted from all the videos created by the previous script with "makeframes.sh". 
Currently sorting into 3 categories must be done by hand starting from the the timestamp given in the notes.



### Initiate transfer learning
Go to the project directory and run:

```
$ ./run.sh  
```

This script installs the ``Inception`` model and initiates the re-training process for the specified image data sets.

The ``training summaries``, ``trained graphs`` and ``trained labels`` will be saved in a folder named ``logs``.

### Classify objects

```
python classify.py image.jpg
```

Where ``image.jpg`` is the input file which is to be classified.

The classifier will output the predictions for each data set.
For getting results from a whole folder, run the image loop script

```
python retrain_predict_loop.py \
-imagepath="/data/TYL/Panama-OBSF_UW_video_cage26/set1_posA"\ 
-modelFullPath="/data/TYL/outputgraph.pb" \
-labelFullPath="/data/TYL/output_labels.txt" \
-outputfile="set1_A.npy"
```
this will return .npy consisting of a matrix of predictions for each image

```
       |label1|label2|label3|
|image1|0.03  |      |      |
| ...  |      |      |      |
|      |      |      |      |
|      |      |      |      |
|     |      |      |      |
```

## Training using the tensorflow script

`retrain.py` uses `--image_dir` as the root folder for training.  
Each sub-folder is named after one of your categories and contains only images from that category.
Script analyzes the sub folders in the image directory, splits them into stable training, testing, and validation sets, and creates an internal data structure
describing the lists of images for each label and their paths. This is done based on filename, as such the same images are used in subsequent invokations of this script

    tensorboard --logdir training_summaries --port 8080 &
    python retrain.py   --bottleneck_dir=bottlenecks   --how_many_training_steps=500   --model_dir=inception  --summaries_dir=training_summaries/basic   --output_graph=retrained_graph.pb   --output_labels=retrained_labels.txt   --image_dir=flower_photos

OR

    python retrain.py \
    --bottleneck_dir=bottlenecks \
    --how_many_training_steps=500 \
    --model_dir=inception \
    --summaries_dir=training_summaries/basic \
    --output_graph=retrained_graph.pb \
    --output_labels=retrained_labels.txt \
    --image_dir=fish_videos



## References
*  Tensor Flow for Poets (https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#4)
*  Tensor Flow Wiki (https://en.wikipedia.org/wiki/TensorFlow)
*  Transfer Learning (https://en.wikipedia.org/wiki/Inductive_transfer)
*  Tensor Flow Image Retraining (https://www.tensorflow.org/versions/master/how_tos/image_retraining/)