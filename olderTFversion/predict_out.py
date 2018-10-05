import os
import numpy as np
import tensorflow as tf
import csv,os
import pandas as pd
import glob

#imagePath = '/data/TYL/Panama-OBSF_UW_video_cage26/Camera_Position_B/Aug 22 2017/postfeed'
modelFullPath = '/data/TYL/tf_files/redo_posB/retrained_graph5.pb'
labelsFullPath = '/data/TYL/tf_files/redo_posB/retrained_labels5.txt'

# FILE NAME TO SAVE TO.
SAVE_TO_CSV = 'tensorflowPred.csv'


def makeCSV():
    global SAVE_TO_CSV
    with open(SAVE_TO_CSV,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])


def makeUniqueDic():
    global SAVE_TO_CSV
    df = pd.read_csv(SAVE_TO_CSV)
    doneID = df['id']
    unique = doneID.unique()
    uniqueDic = {str(key):'' for key in unique} #for faster lookup
    return uniqueDic


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')



def run_inference_on_image():
    answer = []

    imagePath = '/data/TYL/Panama-OBSF_UW_video_cage26/Camera_Position_B/Aug 22 2017/duringfeed/*.jpg'
    testimages=glob.glob(imagePath)
    # Get a list of all files in imagePath directory
    #image_list = tf.gfile.ListDirectory(imagePath)

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for i in range(len(testimages)):
            image_data = tf.gfile.FastGFile(testimages[i], 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n", "") for w in lines]
            pred = labels[top_k[0]]
            with open(SAVE_TO_CSV,'a') as f:
              writer = csv.writer(f)
              writer.writerow([pred])
    return answer
if __name__ == '__main__':
    run_inference_on_image()