
import numpy as np
import tensorflow as tf
import csv,os


imagePath =  '/data/TYL/Panama-OBSF_UW_video_cage26/Camera_Position_B/Aug 22 2017/duringfeed/aug22_duringfeed00001.jpg'
modelFullPath = '/data/TYL/tf_files/redo_posB/retrained_graph5.pb'
labelsFullPath = '/data/TYL/tf_files/redo_posB/retrained_labels5.txt'


# FILE NAME TO SAVE TO.
SAVE_TO_CSV = 'testPred.csv'

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        #predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            with open(SAVE_TO_CSV,'a') as f:
              writer = csv.writer(f)
              writer.writerow([human_string,score])

        
        return answer


if __name__ == '__main__':
    run_inference_on_image()
