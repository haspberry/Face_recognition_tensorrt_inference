#!/usr/sbin/env python
# coding=utf-8

"""
author: jiangqr
file: face_recognition_demo.py
data: 2017.4.11
note: face recognition for image
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
if (sys.version_info[0] == 2):
    reload(sys)
    sys.setdefaultencoding('utf-8')
import time
import cv2
cv_version = 2
if '3' == cv2.__version__[0]:
    cv_version = 3
import detect_face
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import uff
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
import tensorrt as trt



def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def load_label(f):
    labels = {}
    if f is None:
        return None
    while 1:
        line = f.readline()
        if line == '':
            break
        line = line.strip().split(':')
        id, name = line[0], line[1]
        bool_valid = 0 if len(line)>4 else 1
        labels[int(id)] = [name, bool_valid]
        print ('id:{}, label:{}'.format(id, name))
    return labels

#margin = 30
margin = 30
mtcnn_minsize = 160
facenet_imgsize = 160
bbox_minsize = 160
#factor=0.709 
factor = 0.5
#mtcnn_threshold = [0.6, 0.7, 0.7]
mtcnn_threshold = [0.7, 0.8, 0.8]
max_face_num=5
mlp_threshold = 0.8
  
class FaceRecognition(object):
    """face recognition for image"""
    def __init__(self, facenet_model_file, mlp_model_file, label_file, threshold, gpu_id=0):
        
        # load face labels
        self.facenet = facenet_model_file
        
        if os.path.exists(label_file):
            with open(label_file, 'rb') as f:
                self.labels = load_label(f)
        #load mtcnn,facenet,mlp
        with tf.device('/gpu:1'):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth = True)
            #self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, gpu_options = gpu_options))
            self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, gpu_options = gpu_options))
            self.create_graph()
            self.feature_placeholder = tf.placeholder(tf.float32, shape=(1,128))
            self.embeddings = tf.nn.l2_normalize(self.feature_placeholder, 1, 1e-10)
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
            #self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, None)
#            with gfile.FastGFile(facenet_model_file, 'rb') as f:
#                facenet_graph = tf.GraphDef()
#                facenet_graph.ParseFromString(f.read())
#                tf.import_graph_def(facenet_graph, name='')
            with gfile.FastGFile(mlp_model_file, 'rb') as f:
                mlp_graph = tf.GraphDef()
                mlp_graph.ParseFromString(f.read())
                self.mlp_logits, self.mlp_images_features_placehoder = tf.import_graph_def(mlp_graph, return_elements = ['linear/logits:0','Placeholder:0'])

    def create_graph(self):
        uff_model = uff.from_tensorflow_frozen_model(self.facenet,
                                                     ['InceptionResnetV2/Bottleneck/BatchNorm/Reshape_1'], list_nodes = False)
                                                     
        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        parser = uffparser.create_uff_parser()
        parser.register_input('input_image', (3,160,160),0)
        parser.register_output('InceptionResnetV2/Bottleneck/BatchNorm/Reshape_1')
        
        engine = trt.utils.uff_to_trt_engine(G_LOGGER,uff_model,parser,1,1<<31)
        
        parser.destroy()
        
        runtime = trt.infer.create_infer_runtime(G_LOGGER)
        self.context = engine.create_execution_context()
        
        self.output = np.empty((1,128), dtype = np.float32)
        self.d_input = cuda.mem_alloc(1 * 160 * 160 * 3 * 4)
        self.d_output = cuda.mem_alloc(1 * 128 * 4)
        
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()
        
    def run(self, img):
        result = []
        h, w, d = img.shape
        r = h / 500.
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_img = cv2.resize(rgb_img, (int(w / r), int(h / r)))
        
        bounding_boxes, _ = detect_face.detect_face(r_img, mtcnn_minsize, self.pnet, self.rnet, self.onet, mtcnn_threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            rects=np.empty((0,4), dtype=np.int32)
            img_size = np.asarray(r_img.shape)[0:2]
            det = bounding_boxes[:, 0:4]
            if nrof_faces <= max_face_num:
                valid_bounding_boxes = det
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * \
                        (det[:, 3] - det[:, 1])
                max_indexs = np.argsort(-bounding_box_size, axis=0)[0:max_face_num]
                valid_bounding_boxes = [ det[i] for i in max_indexs ][0:max_face_num]
            for box in valid_bounding_boxes:
                rect = np.zeros(4, dtype=np.int32)
                rect[0] = np.maximum(box[0] - margin / 2, 0)
                rect[1] = np.maximum(box[1] - margin / 2, 0)
                rect[2] = np.minimum(box[2] + margin / 2, img_size[1])
                rect[3] = np.minimum(box[3] + margin / 2, img_size[0])
                rects = np.vstack((rects, np.expand_dims(rect, axis=0)))
            emd_num = rects.shape[0]
        else:
            emd_num = 0
            rects = None
            
        if emd_num:
            aligned_imgs = np.empty((0, facenet_imgsize, facenet_imgsize, 3), dtype=img.dtype)
            for rect in rects:
                for i in range(4):
                    rect[i] = int(r*rect[i])
                cropped = rgb_img[rect[1]:rect[3], rect[0]:rect[2], :]
                scaled = cv2.resize(cropped, (facenet_imgsize, facenet_imgsize))
                aligned_imgs = np.vstack((aligned_imgs, np.expand_dims(scaled, axis=0)))

            #images = np.zeros((aligned_imgs.shape[0], facenet_imgsize, facenet_imgsize, 3))
            processed_image = []
            for i, aligned_img in enumerate(aligned_imgs):
                images_temp = prewhiten(aligned_img)
                images_temp_1 = images_temp.astype('float32')
                images_temp_1 = images_temp_1.transpose((2,0,1))
                images_temp_1 = images_temp_1.ravel()
                processed_image.append(images_temp_1)
                #images[i, :, :, :] = images_temp_1
                
            output_temp = []
            for sub_processed_image in processed_image:
                cuda.memcpy_htod_async(self.d_input, sub_processed_image, self.stream)
                self.context.enqueue(1, self.bindings, self.stream.handle, None)
                cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
                output_temp.append(self.output)
                self.stream.synchronize()
            
            images_feature = []
            for i in output_temp:
                images_feature.append(self.sess.run(self.embeddings, {self.feature_placeholder: i}))
            #embs = self.sess.run(self.facenet_embeddings, {self.facenet_images_placeholder: images, self.facenet_phase_train_placeholder: False})
            
            names = []
            scores = []
            probables = []
            for index,i in enumerate(images_feature):
                probables.append(self.sess.run(self.mlp_logits, {self.mlp_images_features_placehoder: i}))
                #print(index,type(probables[index]),probables[index].shape,probables[index].dtype)
            max_indexs = []
            for probable in probables:
                max_indexs.append(np.argmax(probable, axis=1)[0])
                
            for index, probable in enumerate(probables):
                max_index = max_indexs[index]
                names += [self.labels[max_index]]
                soft_score = np.divide(np.exp(probable[0][max_index]), np.sum(np.exp(probable[0])))
                scores += [soft_score]
            for i, score in enumerate(scores):
                if score < mlp_threshold \
                        or rects[i][2] - rects[i][0] < bbox_minsize \
                        or rects[i][3] - rects[i][1] < bbox_minsize \
                        or not names[i][1]:
                    continue
                text = u'{}:{:.2f}'.format(names[i][0], score)
                # print(text)
                result.append([names[i][0], score, rects[i]])
            if len(result):
                print('******{}'.format(result))
        return result

if __name__ == '__main__':
    #facenet_model_file = '/data/work/Recognition_all/FaceRecognition/data/model/face/4.15_12class_author/face_569_no_beddings.pb'
    face_recognize = FaceRecognition('face_model/face_569_no_beddings.pb','face_model/model.ckpt-24000.pb','face_model/face_569.txt',0.7,1)
    cap = cv2.VideoCapture('/data/huang/face_test/face_video_one/sunhonglei_guanxiaotong.mp4')
    frame = 0
    
    if cap.isOpened():
        ret, photo = cap.read()
        while ret and photo is not None:
            if frame % 10 == 0:
                #photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
                #name, score, box, signal = face_recognize.run(photo)
                result_temp = face_recognize.run(photo)
                #print(len(result_temp))
                if len(result_temp) > 2:
                    name, score = result_temp[0], result_temp[1]
                    if score > 0.7:
                        print(frame,name, score)
            ret, photo = cap.read()
            frame += 1
    #        if frame == 20:
    #            break
       
    cap.release()
