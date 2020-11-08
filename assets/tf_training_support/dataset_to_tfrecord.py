import os
import io
import glob
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import random

from PIL import Image
from object_detection.utils import dataset_util

'''
this script automatically divides dataset into training and evaluation (10% for evaluation)
this scripts also shuffles the dataset before converting it into tfrecords
if u have different structure of dataset (rather than pascal VOC ) u need to change
the paths and names input directories(images and annotation) and output tfrecords names.
(note: this script can be enhanced to use flags instead of changing parameters on code).

default expected directories tree:
dataset-
   -JPEGImages
   -Annotations
    dataset_to_tfrecord.py


to run this script:
$ python dataset_to_tfrecord.py

Copyryght: Original script publicly available at: https://gist.github.com/saghiralfasly
Last modified: M. Russ0, Nov 8th 2020
'''
def create_example(xml_file):
        #process the xml file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_name = root.find('filename').text
        file_name = image_name.encode('utf8')
        size=root.find('size')
        width = int(size[0].text)
        height = int(size[1].text)
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []

        for member in root.findall('object'):
           classes_text.append(member[0].text.encode('utf8'))

           if (member[0].text) == 'Green':
               classes.append(1)
           if (member[0].text) == 'Red':
               classes.append(2)
           if (member[0].text) == 'Yellow':
               classes.append(3)

           xmin.append(float(member[4][0].text) / width)
           ymin.append(float(member[4][1].text) / height)
           xmax.append(float(member[4][2].text) / width)
           ymax.append(float(member[4][3].text) / height)
           difficult_obj.append(0)
           truncated.append(0)
           poses.append('Unspecified'.encode('utf8'))

        #read corresponding image
        full_path = os.path.join('./JPEGImages', '{}'.format(image_name))  #provide the path of images directory
        with tf.io.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
           print(image.format)
           raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        #create TFRecord Example
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(file_name),
            'image/source_id': dataset_util.bytes_feature(file_name),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
        return example

def main(_):
    writer_train = tf.io.TFRecordWriter('train.record')
    writer_test = tf.io.TFRecordWriter('test.record')
    #provide the path to annotation xml files directory
    filename_list=tf.io.match_filenames_once("./Annotations/*.xml")
    init = (tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    sess=tf.compat.v1.Session()
    sess.run(init)
    list=sess.run(filename_list)
    random.shuffle(list)   #shuffle files list
    i=1
    tst=0   #to count number of images for evaluation
    trn=0   #to count number of images for training
    for xml_file in list:
      example = create_example(xml_file)
      if (i%10)==0:  #each 10th file (xml and image) write it for evaluation
         print('Annotation/Inage in test: ' + str(xml_file))
         writer_test.write(example.SerializeToString())
         tst=tst+1
      else:          #the rest for training
         writer_train.write(example.SerializeToString())
         trn=trn+1
      i=i+1
    writer_test.close()
    writer_train.close()
    print('Successfully converted dataset to TFRecord.')
    print('training dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)

if __name__ == '__main__':
    tf.compat.v1.app.run()
