import tensorflow as tf
import torch
import onnx
from onnx_tf.backend import prepare
import os
import numpy as np
from models.Nets import CNNCifar
import torch
from torch import nn
import torch.nn.functional as F
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
args = parser.parse_args([])

# set the value of num_classes manually
args.num_classes = 10

os.environ['CUDA_VISIBLE_DEVICES']='-1'

def torch2Onnx():
    """
    pytorch转onnx
    """
    model_pytorch = CNNCifar(args)
    model_pytorch.load_state_dict(torch.load('results/models/cnn.pt',
                                       map_location='cpu'))
    # 输入placeholder
    dummy_input = torch.randn(64, 3, 32, 32, requires_grad=True)
    dummy_output = model_pytorch(dummy_input)
    print(dummy_output.shape)

    # Export to ONNX format
    torch.onnx.export(model_pytorch, 
                      dummy_input, 
                      'model.onnx', 
                      input_names=['inputs'], 
                      output_names=['outputs'])
    

def onnx2Tensorflow(onnx_model="model.onnx", tf_model="cnn.tf.pb"):
    """
    onnx转tensorflow
    """
    """
    # ---------------------报错-------------------------
    # Load ONNX model and convert to TensorFlow format
    model_onnx = onnx.load(onnx_model)
    tf_rep = prepare(model_onnx)
    # Export model as .pb file
    tf_rep.export_graph(tf_model)
    """
    os.system("onnx-tf convert -i %s -o %s"%(onnx_model, tf_model))

def load_pb(path_to_pb):
    """
    加载pb文件
    """
    # Load the protobuf file
    with open(path_to_pb, 'r') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def loadTFModel():
    """
    加载tensorflow模型测试
    """
    graph = load_pb('cnn.tf.pb/saved_model.pb')
    with tf.Session(graph=graph) as sess:
        # Show tensor names in graph
        for op in graph.get_operations():
            # 获取节点名称
            print(op.values())
        # 获取输入输出节点
        output_tensor = graph.get_tensor_by_name('div_3:0')
        input_tensor = graph.get_tensor_by_name('inputs:0')

        # dummy_input = np.random.randint(0, 1000, (1, 20), dtype=np.int64)
        query1 = np.array([[4158, 7811, 6653,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        query2 = np.array([[9914, 10859, 6907, 8719, 7098, 
                            8861, 4158, 10785, 6299, 1264, 
                            1612, 10285, 6973, 7811, 0,
                            0, 0, 0, 0, 0]],
                           dtype=np.int64)
        output1 = sess.run(output_tensor, 
                          feed_dict={input_tensor: query1})
        output2 = sess.run(output_tensor,
                          feed_dict={input_tensor: query2})
        
        simi = np.dot(output1[0], output2[0])
        print(simi)

torch2Onnx()
onnx2Tensorflow(onnx_model="model.onnx", tf_model="cnn.tf.pb")
load_pb('cnn.tf.pb/saved_model.pb')
loadTFModel()



