
import time

start_time = time.time()

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

width = 299
height = 299
channels = 3

import sys
import tarfile
from six.moves import urllib
import os

print("Point 1 elapsed time is {:.2f}s".format(time.time()-start_time))

TF_MODELS_URL = "http://download.tensorflow.org/models"
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()

def fetch_pretrained_inception_v3(url=INCEPTION_V3_URL, path=INCEPTION_PATH):
    if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "inception_v3.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    inception_tgz = tarfile.open(tgz_path)
    inception_tgz.extractall(path=path)
    inception_tgz.close()
    os.remove(tgz_path)

fetch_pretrained_inception_v3()

print("Point 2 elapsed time is {:.2f}s".format(time.time()-start_time))

import re

CLASS_NAME_REGEX = re.compile(r"^n\d+\s+(.*)\s*$", re.M|re.U)

def load_class_names():
    with open(os.path.join("datasets", "inception", "imagenet_class_names.txt"),"rb") as f:
        countent = f.read().decode("utf-8")
        return CLASS_NAME_REGEX.findall(countent)

class_names = load_class_names()

class_names[:5]

print("Point 3 elapsed time is {:.2f}s".format(time.time()-start_time))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

test_image = mpimg.imread(os.path.join("images", "cnn", "test_image.jpeg"))[:,:,:channels]
#plt.imshow(test_image)
#plt.axis("off")
#plt.show()
X_test = test_image.reshape(-1, height, width, channels)

print("Point 4 elapsed time is {:.2f}s".format(time.time()-start_time))

import tensorflow as tf

print("Point 5 elapsed time is {:.2f}s".format(time.time()-start_time))

from tensorflow.contrib.slim.nets import inception
print("Point 6 elapsed time is {:.2f}s".format(time.time()-start_time))

import tensorflow.contrib.slim as slim
print("Point 7 elapsed time is {:.2f}s".format(time.time()-start_time))

import numpy as np


reset_graph()

X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
with slim.arg_scope(inception.inception_v3_arg_scope()):
    print("Point 7.1 elapsed time is {:.2f}s".format(time.time()-start_time))    
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training = False)
print("Point 7.2 elapsed time is {:.2f}s".format(time.time()-start_time))
predictions = end_points["Predictions"]
print("Point 7.3 elapsed time is {:.2f}s".format(time.time()-start_time))
saver = tf.train.Saver()

print("Point 8 elapsed time is {:.2f}s".format(time.time()-start_time))



with tf.Session() as sess:
    saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
    print("Point 9 elapsed time is {:.2f}s".format(time.time()-start_time))

    predictions_val = predictions.eval(feed_dict={X: X_test})
    print("Point 10 elapsed time is {:.2f}s".format(time.time()-start_time))

most_likely_class_index = np.argmax(predictions_val[0])
print(most_likely_class_index)

print(class_names[most_likely_class_index])

top_5 = np.argpartition(predictions_val[0], -5)[-5:]
top_5 = top_5[np.argsort(predictions_val[0][top_5])]

print("Point 11 elapsed time is {:.2f}s".format(time.time()-start_time))
for i in top_5:
    print("{0}: {1:.2f}%".format(class_names[i], 100 * predictions_val[0][i]))

print("Point 12 elapsed time is {:.2f}s".format(time.time()-start_time))    


