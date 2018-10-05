import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

import os
import importlib
import math
import itertools

from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import applications
from keras.utils.np_utils import to_categorical
import math
from sklearn.metrics import accuracy_score, confusion_matrix
import keras.backend as K

img_width, img_height = 299, 299 #fixed size for InceptionV3 architecture
batch_size = 16


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    confusion_matrix_dir = './confusion_matrix_plots'

    plt.cla()
    plt.figure(figsize=(30,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="#BFD1D4" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def predict():
    # load the class_indices saved in the earlier step
    #class_dictionary = np.load('class_indices.npy').item()



    # add the path to your test image below

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator_test = datagen.flow_from_directory(
        predict_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = False)
    class_dictionary = sorted(generator_test.class_indices)
    num_classes = len(class_dictionary)
    print(class_dictionary)

    nb_samples = len(generator_test.filenames)

    if sys.version_info >= (3, 0):
        predict_size = int(math.ceil(nb_samples / batch_size))
    else:
        predict_size = int((math.ceil(nb_samples / batch_size))+1)


    


    #num_classes = len(generator_test.class_indices)

    # get the bottleneck prediction
    #bottleneck_prediction = model.predict(generator_test)
    class_truth = generator_test.classes
    probabilities = model.predict_generator(generator_test, predict_size, verbose=1)
    class_predicted = probabilities.argmax(axis=-1)
    print('Accuracy {}'.format(accuracy_score(y_true=class_truth, y_pred=class_predicted)))
    cnf_matrix = confusion_matrix(class_truth, class_predicted)

    #print(cnf_matrix)
    np.save( modelname+'cnf_matrix.npy',cnf_matrix)
    
    plot_confusion_matrix(cnf_matrix, class_dictionary, normalize=False)
    plt.savefig( modelname +'_Cnf.png')




if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--test_dir", help="path to test folder for predict")
  #a.add_argument("--cnf_path", help="output path for cnf matrix npy")
  a.add_argument("--model", required=True)
  args = a.parse_args()
  if args.test_dir is None :
    a.print_help()
    sys.exit(1)
  predict_data_dir = args.test_dir

  modelname= args.model
  model = load_model(modelname)
  predict()
