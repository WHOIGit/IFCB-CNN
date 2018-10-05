for i in duringsouth/*.jpg; do
   python3 label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --image=$i
   done
