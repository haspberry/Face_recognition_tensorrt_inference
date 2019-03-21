import tensorflow as tf
from tensorflow.contrib import slim

from tensorflow.python.framework.graph_util import convert_variables_to_constants
#from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

OUTPUT_PB_FILENAME = 'face_569_no_beddings.pb' 

from models import inception_resnet_v2

with tf.Graph().as_default():
    
    image_placeholder = tf.placeholder(tf.float32, shape=(1,160,160,3), name='input_image')
    prelogits, _ = inception_resnet_v2.inference(image_placeholder, 0.8, 
            phase_train=False, bottleneck_layer_size=128, weight_decay=5e-5)
    #embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    model_path = '/data/huang/face/20180323-133617/model-20180323-133617.ckpt-180000'

    # Get the function that initializes the network structure (its variables) with
    # the trained values contained in the checkpoint
    init_fn = slim.assign_from_checkpoint_fn(
        model_path,
        slim.get_model_variables())

    with tf.Session() as sess:
        # Now call the initialization function within the session
        init_fn(sess)
        input_graph_def = sess.graph.as_graph_def()
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        # Convert variables to constants and make sure the placeholder input_image is included
        # in the graph as well as the other neccesary tensors.
        constant_graph = convert_variables_to_constants(sess, sess.graph_def, ["input_image","InceptionResnetV2/Bottleneck/BatchNorm/Reshape_1"])

        tf.train.write_graph(constant_graph, '.', OUTPUT_PB_FILENAME, as_text=False)
