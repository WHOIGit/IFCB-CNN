'''
Using Bottleneck Features for Multi-Class Classification in Keras


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
'''
import gc
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import keras.backend as K
import h5py

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_incept_model_fixed49_2.h5'
train_data_dir = '/data/TYL/plank_49_fixed/train'
validation_data_dir = '/data/TYL/plank_49_fixed/valid'
predict_data_dir = '/data/TYL/plank_49_fixed/test/'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16



def save_bottlebeck_features():

    model = applications.InceptionV3(weights='imagenet', include_top=False)

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(generator, (predict_size_train+1), verbose=1)

    np.save('bottleneck_features_train2.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))
    bottleneck_features_validation = model.predict_generator(
        generator, (predict_size_validation+1), verbose=1)

    np.save('bottleneck_features_validation2.npy',
            bottleneck_features_validation)


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices2.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train2.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation2.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    #model.compile(optimizer='rmsprop',
                 # loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


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
    class_dictionary = generator_test.class_indices
    num_classes = len(class_dictionary)
    model = applications.InceptionV3(weights='imagenet', include_top=False)
    print(len(generator_test.filenames))
    print(generator_test.class_indices)
    print(len(generator_test.class_indices))

    nb_test_samples = len(generator_test.filenames)
    #num_classes = len(generator_test.class_indices)

    # get the bottleneck prediction

    predict_size_predict = int(math.ceil(nb_test_samples / batch_size))

    bottleneck_prediction = model.predict_generator(
        generator_test, (predict_size_predict+1), verbose=1)
    #bottleneck_prediction = model.predict_generator(
    #    generator_test, nb_test_samples // batch_size)
    #bottleneck_prediction = model.predict(generator_test)


    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))



    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    class_truth = generator_test.classes
    probabilities = model.predict(bottleneck_prediction)

    print('Accuracy {}'.format(accuracy_score(y_true=class_truth, y_pred=class_predicted)))
    cnf_matrix = confusion_matrix(class_truth, class_predicted)

    print(cnf_matrix)
    np.save('cnf_matrix_49class_2.npy',cnf_matrix)




save_bottlebeck_features()
train_top_model()
predict()
