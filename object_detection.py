import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor
import time
from scipy.stats import norm
import sys
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('type', 'image', "process image or video")
flags.DEFINE_string('image_input', './examples/dog.jpg', "Default input file")
flags.DEFINE_string('video_input', './examples/driving.mp4', "Default input file")

# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
RFCN_GRAPH_FILE = 'rfcn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'
FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08/frozen_inference_graph.pb'

# Colors (one for each class)
COLOR_LIST = sorted([c for c in ImageColor.colormap.keys()])

#
# Utility funcs
#

def load_class_to_label_map():
  class_to_label_file_name = 'cocostuff-labels.txt'
  with open(class_to_label_file_name) as f:
    lines = f.readlines()
  result = {}
  for line in lines:
    parts = line.split(':')
    class_id = int(parts[0])
    class_label = parts[1].strip()
    result[class_id] = class_label
  return result

class_to_label_map = load_class_to_label_map()

def filter_boxes(min_score, boxes, scores, classes):
  """Return boxes with a confidence >= `min_score`"""
  n = len(classes)
  idxs = []
  for i in range(n):
    if scores[i] >= min_score:
      idxs.append(i)

  filtered_boxes = boxes[idxs, ...]
  filtered_scores = scores[idxs, ...]
  filtered_classes = classes[idxs, ...]
  return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
  """
  The original box coordinate output is normalized, i.e [0, 1].

  This converts it back to the original coordinate based on the image
  size.
  """
  box_coords = np.zeros_like(boxes)
  box_coords[:, 0] = boxes[:, 0] * height
  box_coords[:, 1] = boxes[:, 1] * width
  box_coords[:, 2] = boxes[:, 2] * height
  box_coords[:, 3] = boxes[:, 3] * width
  return box_coords

def draw_boxes(image, boxes, classes, scores, thickness=4):
  """Draw bounding boxes on the image"""
  draw = ImageDraw.Draw(image)
  for i in range(len(boxes)):
    top, left, bot, right = boxes[i, ...]
    class_id = int(classes[i])
    score = scores[i]
    color = COLOR_LIST[class_id]
    if top >= 20:
      text_top = top - 20
    else:
      text_top = 0
    font = ImageFont.truetype('arial.ttf', size=15)
    class_label = "{}, {:.0f}%".format(class_to_label_map[class_id], score*100)
    draw.text((left, text_top), class_label, fill=COLOR_LIST[0], font=font)
    draw.line(
      [(left, top), (left, bot), (right, bot), (right, top), (left, top)],
      width=thickness, fill=color
    )

def load_graph(graph_file):
  """Loads a frozen inference graph"""
  graph = tf.Graph()
  with graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(graph_file, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return graph


detection_graph = load_graph(SSD_GRAPH_FILE)
#detection_graph = load_graph(RFCN_GRAPH_FILE)
#detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

def image_obj_detection(sess, image):
  image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
  (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                      feed_dict={image_tensor: image_np})

  # Remove unnecessary dimensions
  boxes = np.squeeze(boxes)
  scores = np.squeeze(scores)
  classes = np.squeeze(classes)

  confidence_cutoff = 0.55
  # Filter boxes with a confidence score less than `confidence_cutoff`
  boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

  # The current box coordinates are normalized to a range between 0 and 1.
  # This converts the coordinates actual location on the image.
  width, height = image.size
  box_coords = to_image_coords(boxes, height, width)

  # Each class with be represented by a differently colored box
  draw_boxes(image, box_coords, classes, scores)
  return np.array(image)


def process_image(file_name):
  image = Image.open(file_name)
  sess = tf.Session(graph=detection_graph)
  image = image_obj_detection(sess, image)
  output_file_name = './output/' + file_name.split('/')[-1].split('.')[0] + '.png'
  mpimg.imsave(output_file_name, image)
  print('Saved file to {}'.format(output_file_name))
  plt.imshow(image)
  plt.show()
  

def process_video(file_name):
  clip = VideoFileClip('driving.mp4')
  sess = tf.Session(graph=detection_graph)
  pipeline = lambda img: image_obj_detection(sess, Image.fromarray(img))
  new_clip = clip.fl_image(pipeline)
  new_clip.write_videofile('result.mp4')


def main():
  if FLAGS_type == 'image':
    process_image(FLAGS_image_file)
  elif FLAGS_type == 'video':
    process_video(FLAGS_video_file)
  elif
    print('Bad input: check the usage')

if __name__ == "__main__":
  main()


