"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import random
import mlp
from six.moves import xrange
import time


def main(args):
    
#    tf_config = tf.ConfigProto()
#    tf_config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

                 
            raw_paths, raw_labels = facenet.get_image_paths_and_labels(dataset)
            
            random_list = range(0,len(raw_paths))
            random.shuffle(random_list)
            paths = []
            labels = []
            for index in random_list:
                paths.append(raw_paths[index])
                labels.append(raw_labels[index])
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            #emb_array = np.zeros((args.batch_size, embedding_size))
            
             
            mlp_images_placeholder = tf.placeholder(tf.float32, shape=(None,int(embedding_size)))
            mlp_labels_placeholder = tf.placeholder(tf.int32, shape=(None))
            logits = mlp.inference(mlp_images_placeholder, int(embedding_size), args.hidden1, args.hidden2, args.number_class)
            loss = mlp.loss(logits, mlp_labels_placeholder)
            train_op = mlp.training(loss, args.learning_rate)
             
            summary = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
            sess.run(init) 
            
            i = -1
            for step in xrange(1000000):
                start_time = time.time()
                i = i + 1
                if i >= nrof_batches_per_epoch:
                    i = 0
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array = np.zeros((end_index-start_index,embedding_size))
                emb_array[0:end_index-start_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                label_array = labels[start_index:end_index]
                
                _, loss_value = sess.run([train_op, loss], {mlp_images_placeholder: emb_array, mlp_labels_placeholder: label_array})
                duration = time.time() - start_time
    
                # write the summaries and print an overview fairly often.
                print('step {}: loss = {} ({} sec)'.format(step, loss_value, duration))
                if step % 100 == 0:
                    # update the events file.
                    summary_str = sess.run(summary, {mlp_images_placeholder: emb_array, mlp_labels_placeholder: label_array})
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    
                if (step + 1) % 1000 == 0:
                    checkpoint_file = os.path.join(args.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=50)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=1000)
    parser.add_argument('--max_steps', type=int, default=4000, help='number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate.') 
    parser.add_argument('--hidden1', type=int, default=1024, help='number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=512, help='number of units in hidden layer 2.')
    parser.add_argument('--number_class', type=int, default=389, help='the total class.')
    parser.add_argument('--log_dir', type=str, default='/data/facenet-master/logs/fully_connected_feed', help='idirectory to put the log data.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
