
import numpy as np
import tensorflow as tf
import glob
import os

modelFullPath = '/home/tyronelee/Scratch/inception-fishfeed/retrain_results7-15/output_graph.pb'
labelsFullPath = '/home/tyronelee/Scratch/inception-fishfeed/retrain_results7-15/output_labels.txt'
output_filename = 'postfeedsouth_predictions.npy'

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


if __name__ == '__main__':

    imagePath = '/home/tyronelee/Scratch/fishsouth/postfeedsouth/*.jpg'
    testimages=glob.glob(imagePath)
    load_labels(labelsFullPath)
    ## init numpy array to hold all predictions
    all_predictions = np.zeros(shape=(len(testimages),3)) ## 3 categories
    #all_predictions[0,:] = np

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for i in range(len(testimages)):
            image_data1 = tf.gfile.FastGFile(testimages[i], 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data1})
            all_predictions[i,:] = np.squeeze(predictions)
            if i % 100 == 0:
              print(str(i) +' of a total of '+ str(len(testimages)))

    np.save(output_filename, all_predictions)
    print (output_filename + ' saved')
