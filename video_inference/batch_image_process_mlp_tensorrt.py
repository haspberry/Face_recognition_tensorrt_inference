# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:28:15 2017

@author: huangchang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import detect_face
import tensorflow as tf
import numpy as np
import facenet
import cv2
from tensorflow.python.platform import gfile
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
import argparse
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
margin = 44
photo_size = 160
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
images_batch = np.zeros((1,160,160,3))

class image_process():
    
    def __init__(self,args):
        self.facenet = args.facenet
        self.mlp = args.mlp
        self.face_classname_path = args.face_classname_path

        self.face_classname = []
        self.output = []
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.3)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.create_graph()
        
        with tf.device('/gpu:1'):
            self.feature_placeholder = tf.placeholder(tf.float32, shape=(1,128))
            self.embeddings = tf.nn.l2_normalize(self.feature_placeholder, 1, 1e-10)
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)

            with gfile.FastGFile(self.mlp, 'rb') as f:
                graph_def_mlp = tf.GraphDef()
                graph_def_mlp.ParseFromString(f.read())
                self.mlp_softmax, self.mlp_images_features_placehoder = tf.import_graph_def(graph_def_mlp, return_elements=['mlp_softmax:0','Placeholder:0'])

        with open(self.face_classname_path,'r') as f:
            for item in f.readlines():
                self.face_classname.append(item.strip().split(':')[1])
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
        print('here')
        self.stream = cuda.Stream()
        
    def main(self,image):  

        high, wide = image.shape[0:2]
        rate = wide / 500
        new_high = round(high / rate)
        resize_image = cv2.resize(image,(500,int(new_high)))
        
        bounding_boxes, _ = detect_face.detect_face(resize_image, minsize, self.pnet, self.rnet, self.onet,threshold, factor)
        bounding_boxes = rate*bounding_boxes
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
	    
            det = bounding_boxes[:,0:4]
            image_size = np.asarray(image.shape)[0:2]
            if nrof_faces > 1:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                image_center = image_size / 2
                offsets = np.vstack([(det[:,0]+det[:,2])/2-image_center[1], (det[:,1]+det[:,3])/2-image_center[0]])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size - offset_dist_squared*2.0)
                det = det[index,:]
            det = np.squeeze(det)
            bb = np.zeros(4, dtype = np.int32)
            bb[0] = np.maximum(det[0] - margin/2, 0)
            bb[1] = np.maximum(det[1] - margin/2, 0)
            bb[2] = np.minimum(det[2] + margin/2, image_size[1])
            bb[3] = np.minimum(det[3] + margin/2, image_size[0])
            cropped = image[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = cv2.resize(cropped, (photo_size,photo_size), interpolation = cv2.INTER_LINEAR)
            scaled = facenet.prewhiten(scaled)
            
            scaled_temp = scaled.astype('float32')
            scaled_temp_1 = scaled_temp.transpose((2,0,1))
            processed_image = scaled_temp_1.ravel()
            cuda.memcpy_htod_async(self.d_input, processed_image, self.stream)
            self.context.enqueue(1, self.bindings, self.stream.handle, None)
            cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
            self.stream.synchronize()
            
            images_feature = self.sess.run(self.embeddings, {self.feature_placeholder: self.output})
            images_result_softmax = self.sess.run(self.mlp_softmax, {self.mlp_images_features_placehoder: images_feature})
            #print(type(images_result_no_norm),images_result_no_norm.shape,images_result_no_norm.dtype)
            best_index = np.argmax(images_result_softmax, axis=1)[0]
#            temp_mother = np.exp(images_result_no_norm[0])
#            best_score = np.divide(np.exp(images_result_no_norm[0][best_index]), np.sum(temp_mother))
            best_score = images_result_softmax[0][best_index]
            best_class_names = self.face_classname[best_index]
            
            return best_class_names,float(best_score),[bb[0],bb[1],bb[2],bb[3]],1
        else:
            return 'no_face',0,[0,0,0,0],0
            
def parse_arguments(argv):
    parse = argparse.ArgumentParser()
    parse.add_argument('--facenet',type = str, help = 'extract feature in the image',default = 'face_model/face_569_no_beddings.pb')
    parse.add_argument('--mlp',type = str, help = 'classifer through the mlp',default = 'face_model/model.ckpt-24000.pb')
    parse.add_argument('--face_classname_path',type = str, help = 'face classname',default = 'face_model/face_class_name_4_4.txt')
    parse.add_argument('--result_path', type = str, help = 'processed the images saved',default = 'mlp_24k_raw')
    return parse.parse_args(argv)

if __name__ == '__main__':
    img_handle = image_process(parse_arguments(sys.argv[1:]))
    cap = cv2.VideoCapture('/data/huang/face_test/face_video_one/sunhonglei_guanxiaotong.mp4')
    frame = 0
    
    if cap.isOpened():
        ret, photo = cap.read()
        while ret and photo is not None:
            if frame % 10 == 0:
                photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
                name, score, box, signal = img_handle.main(photo)
                #photo = cv2.resize(photo, (160, 160))
                if signal == 1:
                    if score > 0.7:
                        print(name, score)
            ret, photo = cap.read()
            frame += 1
    #        if frame == 20:
    #            break
       
    cap.release()
