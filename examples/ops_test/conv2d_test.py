# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Conv2d_shape_4_2_1_1_float32_PaddingTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_1_1_float32_PaddingTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 1, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 1, 3]
    weight_data = np.array([1,  2,  3,  4,  #first 2x2 filter
                            -1, 1,  -1, 1,  # second 2x2 filter
                            -1, -1, 1,  1,  # third 2x2 filter
                            ]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [3,]
    bias_data = np.array([1, 2, 3]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 2, 3, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="SAME", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # export graph
    # timvx_engine.export_graph("./conv2d_graph.json", "./conv2d_weight.bin")

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 2, 4]
    input_data_list = [1, 1, 1, 1,  # row = 1
                       2, 2, 3, 2]   # row = 2
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 3, 2, 4]
    golden_data_list = [18, 22, 21, 8, 7, 9, 8, 3, 2, 3, 1, -1, # first channel
                        2, 3, 1, 0, 5, 6, 6, 4, -1, -2, -2, 1] # second channel
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_2_2_2_float32_PointwiseTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_2_2_float32_PointwiseTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 2, 2] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [1, 1, 2, 1] # whio
    weight_data = np.array([1, 2]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_data = np.array([0,]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 2, 1, 2]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="SAME", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [2, 2, 2, 4]
    input_data_list = [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1,
      0.5, 1, 1.5, 2, 0.5, 1, 1.5, 2, 0.5, 1, 1.5, 2, 0.5, 1, 1.5, 2]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 1, 2, 4]
    golden_data_list = [1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3,
                        1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_2_1_2_float32_SimpleTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_1_2_float32_SimpleTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 1, 2] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 1, 3] # whio
    weight_data = np.array([1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [3,]
    bias_data = np.array([1, 2, 3]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 1, 3, 2]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="SAME", stride=[2, 2], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [2, 1, 2, 4]
    input_data_list = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 3, 1, 2]
    golden_data_list = [18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_2_2_2_float32_SimpleChannelsTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_2_2_float32_SimpleChannelsTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 2, 2] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 2, 3] # whio
    weight_data = np.array([1,  2, 3,  4, 1,  2,  3, 4, -1, 1,  -1, 1,
                            -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1,  1]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [3,]
    bias_data = np.array([1, 2, 3]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 1, 3, 2]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="SAME", stride=[2, 2], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [2, 2, 2, 4]
    input_data_list = [0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
                        0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 3, 1, 2]
    golden_data_list = [18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_6_3_1_1_float32_SimpleAnisotropicStridesTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_6_3_1_1_float32_SimpleAnisotropicStridesTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [6, 3, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 1, 1] # whio
    weight_data = np.array([1,  2, 3,  4]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_data = np.array([-1]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 2, 1, 1]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[3, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 3, 6]
    input_data_list = [3,  2,  1,  -1, -2, -3, 4,  3,  2,
                       -2, -3, -4, 5,  4,  3,  -3, -4, -5]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 2, 2]
    golden_data_list = [30, -24, 40, -34]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_3_1_1_float32_HandCalculatedTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_3_1_1_float32_HandCalculatedTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 3, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whio
    weight_data = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_data = np.array([0]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 3, 1, 1]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="SAME", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 3, 4]
    input_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 3, 4]
    golden_data_list = [105, 150, 183, 95,  235, 312,
                        357, 178, 187, 234, 261, 121]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_3_1_1_float32_HandCalculatedConstFilterTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_3_1_1_float32_HandCalculatedConstFilterTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 3, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whio
    weight_data = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_data = np.array([0]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 3, 1, 1]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="SAME", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 3, 4]
    input_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 3, 4]
    golden_data_list = [105, 150, 183, 95,  235, 312,
                        357, 178, 187, 234, 261, 121]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_3_1_1_float32_HandCalculatedBiasTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_3_1_1_float32_HandCalculatedBiasTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 3, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whio
    weight_data = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_data = np.array([10]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 3, 1, 1]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="SAME", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 3, 4]
    input_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 3, 4]
    golden_data_list = [115, 160, 193, 105, 245, 322,
                        367, 188, 197, 244, 271, 131]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_3_1_1_float32_HandCalculatedValidTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_3_1_1_float32_HandCalculatedValidTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 3, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whio
    weight_data = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_data = np.array([0]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 1, 1, 1]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 3, 4]
    input_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 1, 2]
    golden_data_list = [312, 357]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_2_2_2_float32_DisabledPointwiseMultifilterTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_2_2_float32_DisabledPointwiseMultifilterTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 2, 2] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [1, 1, 2, 2] # whio
    weight_data = np.array([1, 2, 2, 3]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [2,]
    bias_data = np.array([0, 0]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 2, 2, 2]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [2, 2, 2, 4]
    input_data_list = [0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
                       0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 2, 2, 4]
    golden_data_list = [1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3, 2.5, 2.5, 2.5, 2.5, 5,   5, 5,   5,
                        1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6, 2.5, 5,   7.5, 10,  2.5, 5, 7.5, 10]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_9_9_1_1_float32_SimpleDilationTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_9_9_1_1_float32_SimpleDilationTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [9, 9, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whio
    weight_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_data = np.array([0]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [3, 3, 1, 1]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[1, 1], dilation=[3, 3], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 9, 9]
    input_data_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                       0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 3, 3]
    golden_data_list = [5, 5, 5, 5, 5, 5, 5, 5, 5]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_2_1_2_float32_StrideTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_1_2_float32_StrideTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 1, 2] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 1, 3] # whio
    weight_data = np.array([1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [3,]
    bias_data = np.array([1, 2, 3]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [3, 1, 3, 2]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [2, 1, 2, 4]
    input_data_list = [1, 1, 1, 1, 2, 2, 3, 2,
                       1, 2, 3, 4, 1, 2, 4, 4]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 3, 1, 3]
    golden_data_list = [18, 22, 21, 2, 3, 1, 5, 6, 6,
                        17, 31, 40, 4, 5, 3, 3, 4, 4]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_2_1_2_float32_InputAndFilterSameWidthHeightTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_1_2_float32_InputAndFilterSameWidthHeightTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 1, 2] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [4, 2, 1, 1] # whio
    weight_data = np.array([1, 2, 3, 4, -1, -1, 1, 1]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_data = np.array([0]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [1, 1, 1, 2]  # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[1, 1], dilation=[0, 0], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [2, 1, 2, 4]
    input_data_list = [1, 1, 1, 1, 2, 2, 2, 2,
                       1, 2, 3, 4, 1, 2, 3, 4]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 1, 1, 1]
    golden_data_list = [10, 34]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_2_1_2_uint8_QuantizedTest1():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_1_2_uint8_QuantizedTest1")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -63.5
    input_max = 64
    weight_min = -63.5
    weight_max = 64
    output_min = -127
    output_max = 128

    input_scale, input_zp = quantizationParams(input_min, input_max, np.uint8)
    weight_scale, weight_zp = quantizationParams(weight_min, weight_max, np.uint8)
    bias_scale, bias_zp = input_scale * weight_scale, 0
    output_scale, output_zp = quantizationParams(output_min, output_max, np.uint8)

    input_name = "input"
    input_tensor_shape = [4, 2, 1, 2] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 1, 3] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = [weight_scale]
    weight_quant_info["zero_points"] = [weight_zp]
    weight_quant_info["channel_dim"] = 2
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data_f = np.array([1,  2, 3,  4,  -1, 1,
                              -1, 1, -1, -1, 1,  1]).reshape(weight_tensor_shape).astype(np.float32)
    weight_data = quantize(weight_data_f, weight_scale, weight_zp, np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [3,]
    bias_quant_info = {}
    bias_quant_info["scales"] = [bias_scale]
    bias_quant_info["zero_points"] = [bias_zp]
    bias_quant_info["channel_dim"] = 2
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    bias_data_f = np.array([1, 2, 3]).reshape(bias_tensor_shape).astype(np.float32)
    bias_data = quantize(bias_data_f, bias_scale, bias_zp, np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 1, 3, 2]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[2, 2], dilation=[1, 1], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [2, 1, 2, 4]
    input_data_list = [1, 1, 1, 1, 2, 2, 2, 2,
                       1, 2, 3, 4, 1, 2, 3, 4]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [2, 3, 1, 2]
    golden_data_list = [18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    golden_data = quantize(golden_data_f, output_scale, output_zp, np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_4_2_1_2_uint8_QuantizedTest2():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_4_2_1_2_uint8_QuantizedTest2")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -128.5
    input_max = 128
    weight_min = -128.5
    weight_max = 128
    output_min = -127
    output_max = 128

    input_scale, input_zp = quantizationParams(input_min, input_max, np.uint8)
    weight_scale, weight_zp = quantizationParams(weight_min, weight_max, np.uint8)
    bias_scale, bias_zp = input_scale * weight_scale, 0
    output_scale, output_zp = quantizationParams(output_min, output_max, np.uint8)

    input_name = "input"
    input_tensor_shape = [4, 2, 1, 2] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 1, 3] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = [weight_scale]
    weight_quant_info["zero_points"] = [weight_zp]
    weight_quant_info["channel_dim"] = 2
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data_f = np.array([1,  2, 3,  4,  -1, 1,
                              -1, 1, -1, -1, 1,  1]).reshape(weight_tensor_shape).astype(np.float32)
    weight_data = quantize(weight_data_f, weight_scale, weight_zp, np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [3,]
    bias_quant_info = {}
    bias_quant_info["scales"] = [bias_scale]
    bias_quant_info["zero_points"] = [bias_zp]
    bias_quant_info["channel_dim"] = 2
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    bias_data_f = np.array([1, 2, 3]).reshape(bias_tensor_shape).astype(np.float32)
    bias_data = quantize(bias_data_f, bias_scale, bias_zp, np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 1, 3, 2]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[2, 2], dilation=[1, 1], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [2, 1, 2, 4]
    input_data_list = [1, 1, 1, 1, 2, 2, 2, 2,
                       1, 2, 3, 4, 1, 2, 3, 4]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [2, 3, 1, 2]
    golden_data_list = [18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    golden_data = quantize(golden_data_f, output_scale, output_zp, np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_6_3_1_1_uint8_AnisotropicStridesQuantizedTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_6_3_1_1_uint8_AnisotropicStridesQuantizedTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -63.5
    input_max = 64
    weight_min = -63.5
    weight_max = 64
    output_min = -127
    output_max = 128

    input_scale, input_zp = quantizationParams(input_min, input_max, np.uint8)
    weight_scale, weight_zp = quantizationParams(weight_min, weight_max, np.uint8)
    bias_scale, bias_zp = input_scale * weight_scale, 0
    output_scale, output_zp = quantizationParams(output_min, output_max, np.uint8)

    input_name = "input"
    input_tensor_shape = [6, 3, 1, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 1, 1] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = [weight_scale]
    weight_quant_info["zero_points"] = [weight_zp]
    weight_quant_info["channel_dim"] = 2
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data_f = np.array([1, 2, 3, 4]).reshape(weight_tensor_shape).astype(np.float32)
    weight_data = quantize(weight_data_f, weight_scale, weight_zp, np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_quant_info = {}
    bias_quant_info["scales"] = [bias_scale]
    bias_quant_info["zero_points"] = [bias_zp]
    bias_quant_info["channel_dim"] = 2
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    bias_data_f = np.array([-1]).reshape(bias_tensor_shape).astype(np.float32)
    bias_data = quantize(bias_data_f, bias_scale, bias_zp, np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 2, 1, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[3, 1], dilation=[1, 1], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 3, 6]
    input_data_list = [3,  2,  1,  -1, -2, -3, 4,  3,  2,
                       -2, -3, -4, 5,  4,  3,  -3, -4, -5]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=False, want_float=True)

    # compare gloden data with output data
    output_np_shape = [1, 1, 2, 2]
    golden_data_list = [30, -24, 40, -34]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    golden_data = quantize(golden_data_f, output_scale, output_zp, np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_9_9_1_1_uint8_DilationQuantizedTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_9_9_1_1_uint8_DilationQuantizedTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -128
    input_max = 127
    weight_min = -128
    weight_max = 127
    output_min = 0
    output_max = 255

    input_scale, input_zp = quantizationParams(input_min, input_max, np.uint8)
    weight_scale, weight_zp = quantizationParams(weight_min, weight_max, np.uint8)
    bias_scale, bias_zp = input_scale * weight_scale, 0
    output_scale, output_zp = quantizationParams(output_min, output_max, np.uint8)

    input_name = "input"
    input_tensor_shape = [9, 9, 1, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = [weight_scale]
    weight_quant_info["zero_points"] = [weight_zp]
    weight_quant_info["channel_dim"] = 2
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data_f = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(weight_tensor_shape).astype(np.float32)
    weight_data = quantize(weight_data_f, weight_scale, weight_zp, np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_quant_info = {}
    bias_quant_info["scales"] = [bias_scale]
    bias_quant_info["zero_points"] = [bias_zp]
    bias_quant_info["channel_dim"] = 2
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    bias_data_f = np.array([0]).reshape(bias_tensor_shape).astype(np.float32)
    bias_data = quantize(bias_data_f, bias_scale, bias_zp, np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [3, 3, 1, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[1, 1], dilation=[3, 3], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 9, 9]
    input_data_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                       0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=False, want_float=True)

    # compare gloden data with output data
    output_np_shape = [1, 1, 3, 3]
    golden_data_list = [5, 5, 5, 5, 5, 5, 5, 5, 5]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    golden_data = quantize(golden_data_f, output_scale, output_zp, np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_3_2_2_1_int8_QuantizedPerTensorTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_3_2_2_1_int8_QuantizedPerTensorTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -63.5
    input_max = 64
    weight_min = -63.5
    weight_max = 64
    output_min = -63.5
    output_max = 64

    input_scale, input_zp = quantizationParams(input_min, input_max, np.int8)
    weight_scale, weight_zp = 1, 0
    bias_scale, bias_zp = input_scale * weight_scale, 0
    output_scale, output_zp = quantizationParams(output_min, output_max, np.int8)

    input_name = "input"
    input_tensor_shape = [3, 2, 2, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "INT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 2, 2] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = [weight_scale]
    weight_quant_info["zero_points"] = [weight_zp]
    weight_quant_info["channel_dim"] = 2
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data_f = np.array([1, 3, 3, 5, 2, 4, 4, 6, 7, 5, 3, 1, 8, 6, 4, 2]).reshape(weight_tensor_shape).astype(np.float32)
    weight_data = quantize(weight_data_f, weight_scale, weight_zp, np.int8)
    assert timvx_engine.create_tensor(weight_name, "INT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [2,]
    bias_quant_info = {}
    bias_quant_info["scales"] = [bias_scale]
    bias_quant_info["zero_points"] = [bias_zp]
    bias_quant_info["channel_dim"] = 2
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    bias_data_f = np.array([3, -2]).reshape(bias_tensor_shape).astype(np.float32)
    bias_data = quantize(bias_data_f, bias_scale, bias_zp, np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 1, 2, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "INT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[1, 1], dilation=[1, 1], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 2, 3]
    input_data_list = [3, 1,  -2, 4, 2,  -3,
                       2, -1, -3, 3, -2, -4]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.int8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=False, want_float=True)

    # compare gloden data with output data
    output_np_shape = [1, 2, 1, 2]
    golden_data_list = [31, -57, 56, -44]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    golden_data = quantize(golden_data_f, output_scale, output_zp, np.int8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv2d_shape_3_2_2_1_int8_QuantizedPerChannelTest():
    # create graph
    timvx_engine = Engine("test_Conv2d_shape_3_2_2_1_int8_QuantizedPerChannelTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -63.5
    input_max = 64
    weight_min = 0
    weight_max = 0
    output_min = -63.5
    output_max = 64

    input_scale, input_zp = quantizationParams(input_min, input_max, np.int8)
    weight_scale, weight_zp = (1, 2), (0, 0)
    bias_scale, bias_zp = (input_scale * weight_scale[0], input_scale * weight_scale[1]), (0, 0)
    output_scale, output_zp = quantizationParams(output_min, output_max, np.int8)

    input_name = "input"
    input_tensor_shape = [3, 2, 2, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "INT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2, 2, 2, 2] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = list(weight_scale)
    weight_quant_info["zero_points"] = list(weight_zp)
    weight_quant_info["channel_dim"] = 3
    weight_quant_info["quant_type"] = "SYMMETRIC_PER_CHANNEL"
    # weight_data_f = np.array([1, 3, 3, 5, 2, 4, 4, 6, 7, 5, 3, 1, 8, 6, 4, 2]).reshape(weight_tensor_shape).astype(np.float32)
    # weight_data = quantize(weight_data_f, weight_scale, weight_zp, np.uint8)
    weight_data = np.array([1, 3, 3, 5, 2, 4, 4, 6, 4, 3, 2, 1, 4, 3, 2, 1]).reshape(weight_tensor_shape).astype(np.int8)
    assert timvx_engine.create_tensor(weight_name, "INT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [2,]
    bias_quant_info = {}
    bias_quant_info["scales"] = list(bias_scale)
    bias_quant_info["zero_points"] = list(bias_zp)
    bias_quant_info["channel_dim"] = 0
    bias_quant_info["quant_type"] = "SYMMETRIC_PER_CHANNEL"
    # bias_data_f = np.array([3, -2]).reshape(bias_tensor_shape).astype(np.float32)
    # bias_data = quantize(bias_data_f, bias_scale, bias_zp, np.int32)
    bias_data = np.array([6, -2]).reshape(bias_tensor_shape).astype(np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2, 1, 2, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "INT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, padding="VALID", stride=[1, 1], dilation=[1, 1], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 2, 3]
    input_data_list = [3, 1,  -2, 4, 2,  -3,
                       2, -1, -3, 3, -2, -4]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.int8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=False, want_float=True)

    # compare gloden data with output data
    output_np_shape = [1, 2, 1, 2]
    golden_data_list = [31, -57, 64, -46]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    golden_data = quantize(golden_data_f, output_scale, output_zp, np.int8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["Conv2d_shape_4_2_1_1_float32_PaddingTest"] = test_Conv2d_shape_4_2_1_1_float32_PaddingTest
test_func_map["Conv2d_shape_4_2_2_2_float32_PointwiseTest"] = test_Conv2d_shape_4_2_2_2_float32_PointwiseTest
test_func_map["Conv2d_shape_4_2_1_2_float32_SimpleTest"] = test_Conv2d_shape_4_2_1_2_float32_SimpleTest
test_func_map["Conv2d_shape_4_2_2_2_float32_SimpleChannelsTest"] = test_Conv2d_shape_4_2_2_2_float32_SimpleChannelsTest
test_func_map["Conv2d_shape_6_3_1_1_float32_SimpleAnisotropicStridesTest"] = test_Conv2d_shape_6_3_1_1_float32_SimpleAnisotropicStridesTest
test_func_map["Conv2d_shape_4_3_1_1_float32_HandCalculatedTest"] = test_Conv2d_shape_4_3_1_1_float32_HandCalculatedTest
test_func_map["Conv2d_shape_4_3_1_1_float32_HandCalculatedConstFilterTest"] = test_Conv2d_shape_4_3_1_1_float32_HandCalculatedConstFilterTest
test_func_map["Conv2d_shape_4_3_1_1_float32_HandCalculatedBiasTest"] = test_Conv2d_shape_4_3_1_1_float32_HandCalculatedBiasTest
test_func_map["Conv2d_shape_4_3_1_1_float32_HandCalculatedValidTest"] = test_Conv2d_shape_4_3_1_1_float32_HandCalculatedValidTest
test_func_map["Conv2d_shape_4_2_2_2_float32_DisabledPointwiseMultifilterTest"] = test_Conv2d_shape_4_2_2_2_float32_DisabledPointwiseMultifilterTest
test_func_map["Conv2d_shape_9_9_1_1_float32_SimpleDilationTest"] = test_Conv2d_shape_9_9_1_1_float32_SimpleDilationTest
test_func_map["Conv2d_shape_4_2_1_2_float32_StrideTest"] = test_Conv2d_shape_4_2_1_2_float32_StrideTest
test_func_map["Conv2d_shape_4_2_1_2_float32_InputAndFilterSameWidthHeightTest"] = test_Conv2d_shape_4_2_1_2_float32_InputAndFilterSameWidthHeightTest
test_func_map["Conv2d_shape_4_2_1_2_uint8_QuantizedTest1"] = test_Conv2d_shape_4_2_1_2_uint8_QuantizedTest1
test_func_map["Conv2d_shape_4_2_1_2_uint8_QuantizedTest2"] = test_Conv2d_shape_4_2_1_2_uint8_QuantizedTest2
test_func_map["Conv2d_shape_6_3_1_1_uint8_AnisotropicStridesQuantizedTest"] = test_Conv2d_shape_6_3_1_1_uint8_AnisotropicStridesQuantizedTest
test_func_map["Conv2d_shape_9_9_1_1_uint8_DilationQuantizedTest"] = test_Conv2d_shape_9_9_1_1_uint8_DilationQuantizedTest
test_func_map["Conv2d_shape_3_2_2_1_int8_QuantizedPerTensorTest"] = test_Conv2d_shape_3_2_2_1_int8_QuantizedPerTensorTest
test_func_map["Conv2d_shape_3_2_2_1_int8_QuantizedPerChannelTest"] = test_Conv2d_shape_3_2_2_1_int8_QuantizedPerChannelTest

def test_conv2d_op():
    test_result = {}
    for key, value in test_func_map.items():
        try:
            print("[ RUN      ] test_{}".format(key))
            test_func_map[key]()
            test_result[key] = "success"
            print("[       OK ]")
        except Exception as e:
            test_result[key] = "fail"
            print("[       FAIL ]")
            # print("exception:\n{}".format(e))
            traceback.print_exc()
    return test_result

if __name__ == "__main__":
    test_conv2d_op()