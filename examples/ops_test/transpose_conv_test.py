# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_TransposeConv2d_shape_4_4_1_1_float32_SimpleTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_1_1_float32_SimpleTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 4, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whc1 same as depthwise convolution
    weight_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [4, 4, 1, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "SAME"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 4, 4]
    input_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 4, 4]
    golden_data_list = [29, 62, 83, 75, 99, 192, 237, 198, 207, 372, 417, 330, 263, 446, 485, 365]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_TransposeConv2d_shape_4_4_2_1_float32_SameTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_2_1_float32_SameTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 4, 2, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 2, 1] # whc1 same as depthwise convolution
    weight_data = np.array([1, 3, 5, 7, 9,  11, 13, 15, 17,
                            2, 4, 6, 8, 10, 12, 14, 16, 18]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [4, 4, 1, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "SAME"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 4, 4]
    input_data_list = [1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21,
                       23, 25, 27, 29, 31, 2,  4,  6,  8,  10, 12,
                       14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 4, 4]
    golden_data_list = [184,  412,  568,  528,  678,  1347, 1689, 1434,
                        1494, 2715, 3057, 2442, 1968, 3352, 3652, 2760]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_TransposeConv2d_shape_4_4_2_1_float32_ValidTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_2_1_float32_ValidTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 4, 2, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 2, 1] # whc1 same as depthwise convolution
    weight_data = np.array([1, 3, 5, 7, 9,  11, 13, 15, 17,
                            2, 4, 6, 8, 10, 12, 14, 16, 18]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [6, 6, 1, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "VALID"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 4, 4]
    input_data_list = [1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21,
                       23, 25, 27, 29, 31, 2,  4,  6,  8,  10, 12,
                       14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 6, 6]
    golden_data_list = [5, 22, 59, 101, 114, 83, 52, 184, 412, 568, 528, 344,
                        237, 678, 1347, 1689, 1434, 879, 597, 1494, 2715, 3057, 2442, 1431,
                        856, 1968, 3352, 3652, 2760, 1548, 689, 1534, 2543, 2729, 2010, 1103]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_TransposeConv2d_shape_2_2_1_1_float32_StrideTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_2_2_1_1_float32_StrideTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [2, 2, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whc1 same as depthwise convolution
    weight_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [5, 5, 1, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "VALID"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [2, 2]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 2, 2]
    input_data_list = [1, 2, 3, 4]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 5, 5]
    golden_data_list = [1,  2,  5,  4,  6,  4,  5,  14, 10,
                        12, 10, 14, 36, 24, 30, 12, 15, 34,
                        20, 24, 21, 24, 55, 32, 36]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_TransposeConv2d_shape_2_2_1_1_float32_ChannelTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_2_2_1_1_float32_ChannelTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [2, 2, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 2] # whc1 same as depthwise convolution
    weight_data = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [5, 5, 2, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "VALID"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [2, 2]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 2, 2]
    input_data_list = [1, 2, 3, 4]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 5, 5]
    golden_data_list = [1, 3, 7, 6, 10, 7, 9, 25, 18, 22, 16, 24, 62, 42, 54, 21, 27,
                        61, 36, 44, 39, 45, 103, 60, 68, 2, 4,  10, 8, 12, 8, 10, 28, 20,
                        24, 20, 28, 72, 48, 60, 24, 30, 68, 40, 48, 42, 48, 110, 64, 72]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_TransposeConv2d_shape_2_1_1_1_float32_AccuracyTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_2_1_1_1_float32_AccuracyTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [2, 1, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whc1 same as depthwise convolution
    weight_data = np.array([9, 5, 6, 9, 8, 5, 3, 1, 4]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [4, 3, 1, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "SAME"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [3, 3]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 1, 2]
    input_data_list = [323, 521]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 3, 4]
    golden_data_list = [1615.0, 1938.0, 4689.0, 2605.0, 2584.0, 1615.0,
                        4689.0, 4168.0, 323.0,  1292.0, 1563.0, 521.0]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_TransposeConv2d_shape_2_2_1_1_float32_BiasChannelTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_2_2_1_1_float32_BiasChannelTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [2, 2, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 2] # whc1 same as depthwise convolution
    weight_data = np.array([1, 3, 5, 7, 9,  11, 13, 15, 17,
                            2, 4, 6, 8, 10, 12, 14, 16, 18]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [weight_tensor_shape[3]] # whc1 same as depthwise convolution
    bias_data = np.array([3, 4]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [5, 5, 2, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    padding = "VALID"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [2, 2]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 2, 2]
    input_data_list = [1, 2, 3, 4]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 5, 5]
    golden_data_list = [4,  6,  10, 9,  13, 10,  12, 28, 21, 25, 19, 27, 65, 45,  57, 24, 30,
                        64, 39, 47, 42, 48, 106, 63, 71, 6,  8,  14, 12, 16, 12,  14, 32, 24,
                        28, 24, 32, 76, 52, 64,  28, 34, 72, 44, 52, 46, 52, 114, 68, 76]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_TransposeConv2d_shape_4_4_1_1_uint8_QuantizedTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_1_1_uint8_QuantizedTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -63.5
    input_max = 64
    weight_min = -63.5
    weight_max = 64
    output_min = -508
    output_max = 512

    input_scale, input_zp = quantizationParams(input_min, input_max, np.uint8)
    weight_scale, weight_zp = quantizationParams(weight_min, weight_max, np.uint8)
    output_scale, output_zp = quantizationParams(output_min, output_max, np.uint8)

    input_name = "input"
    input_tensor_shape = [4, 4, 1, 1] # whcn
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
    weight_data = np.array([129, 131, 133, 135, 137, 139, 141, 143, 145]).reshape(weight_tensor_shape).astype(np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, 
        np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [4, 4, 1, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "SAME"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 4, 4]
    input_data_list = [1, 2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [1, 1, 4, 4]
    golden_data_list = [28,  64,  84,  76,  100, 192, 236, 200,
                        208, 372, 416, 332, 264, 448, 484, 364]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    output_data_f = dequantize(output_data[0], output_scale, output_zp)
    assert np.allclose(golden_data_f, output_data_f, atol=output_scale), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data_f, output_data_f)

def test_TransposeConv2d_shape_4_4_2_1_uint8_QuantizedTwoFiltersTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_2_1_uint8_QuantizedTwoFiltersTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -63.5
    input_max = 64
    weight_min = -63.5
    weight_max = 64
    output_min = -4064
    output_max = 4096

    input_scale, input_zp = quantizationParams(input_min, input_max, np.uint8)
    weight_scale, weight_zp = quantizationParams(weight_min, weight_max, np.uint8)
    output_scale, output_zp = quantizationParams(output_min, output_max, np.uint8)

    input_name = "input"
    input_tensor_shape = [4, 4, 2, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 2, 1] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = [weight_scale]
    weight_quant_info["zero_points"] = [weight_zp]
    weight_quant_info["channel_dim"] = 2
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data = np.array([129, 133, 137, 141, 145, 149,
                            153, 157, 161, 131, 135, 139,
                            143, 147, 151, 155, 159, 163]).reshape(weight_tensor_shape).astype(np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, 
        np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [4, 4, 1, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "SAME"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 4, 4]
    input_data_list = [1, 3, 5, 7, 9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                       2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [1, 1, 4, 4]
    golden_data_list = [192,  416,  576,  544,  672,  1344,
                        1696, 1440, 1504, 2720, 3072, 2432,
                        1984, 3360, 3648, 2752]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    output_data_f = dequantize(output_data[0], output_scale, output_zp)
    assert np.allclose(golden_data_f, output_data_f, atol=output_scale), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data_f, output_data_f)

def test_TransposeConv2d_shape_4_4_2_1_uint8_QuantizedValidTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_2_1_uint8_QuantizedValidTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -63.5
    input_max = 64
    weight_min = -63.5
    weight_max = 64
    output_min = -4064
    output_max = 4096

    input_scale, input_zp = quantizationParams(input_min, input_max, np.uint8)
    weight_scale, weight_zp = quantizationParams(weight_min, weight_max, np.uint8)
    output_scale, output_zp = quantizationParams(output_min, output_max, np.uint8)

    input_name = "input"
    input_tensor_shape = [4, 4, 2, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 2, 1] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = [weight_scale]
    weight_quant_info["zero_points"] = [weight_zp]
    weight_quant_info["channel_dim"] = 2
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data = np.array([129, 133, 137, 141, 145, 149,
                            153, 157, 161, 131, 135, 139,
                            143, 147, 151, 155, 159, 163]).reshape(weight_tensor_shape).astype(np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, 
        np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [6, 6, 1, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", ]
    op_outputs = ["output", ]
    padding = "VALID"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 4, 4]
    input_data_list = [1, 3, 5, 7, 9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                       2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [1, 1, 6, 6]
    golden_data_list = [0,   32,   64,   96,   128,  96,   64,  192,  416,  576,  544,  352,
                        224, 672,  1344, 1696, 1440, 864,  608, 1504, 2720, 3072, 2432, 1440,
                        864, 1984, 3360, 3648, 2752, 1536, 704, 1536, 2528, 2720, 2016, 1088]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    output_data_f = dequantize(output_data[0], output_scale, output_zp)
    assert np.allclose(golden_data_f, output_data_f, atol=output_scale), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data_f, output_data_f)

def test_TransposeConv2d_shape_4_4_1_1_uint8_QuantizedBiasTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_1_1_uint8_QuantizedBiasTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_min = -63.5
    input_max = 64
    weight_min = -63.5
    weight_max = 64
    output_min = -508
    output_max = 512

    input_scale, input_zp = quantizationParams(input_min, input_max, np.uint8)
    weight_scale, weight_zp = quantizationParams(weight_min, weight_max, np.uint8)
    bias_scale, bias_zp = input_scale * weight_scale, 0
    output_scale, output_zp = quantizationParams(output_min, output_max, np.uint8)

    input_name = "input"
    input_tensor_shape = [4, 4, 1, 1] # whcn
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
    weight_data = np.array([129, 131, 133, 135, 137, 139, 141, 143, 145]).reshape(weight_tensor_shape).astype(np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, 
        np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [weight_tensor_shape[3]]
    bias_quant_info = {}
    bias_quant_info["scales"] = [bias_scale]
    bias_quant_info["zero_points"] = [bias_zp]
    bias_quant_info["channel_dim"] = 2
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    bias_data_f = np.array([1]).reshape(bias_tensor_shape).astype(np.float32)
    bias_data = quantize(bias_data_f, bias_scale, bias_zp, np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 4, 1, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", "bias", ]
    op_outputs = ["output", ]
    padding = "SAME"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 4, 4]
    input_data_list = [1, 2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [1, 1, 4, 4]
    golden_data_list = [32,  64,  84,  76,  100, 192, 240, 200,
                        208, 372, 420, 332, 264, 448, 488, 368]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    output_data_f = dequantize(output_data[0], output_scale, output_zp)
    assert np.allclose(golden_data_f, output_data_f, atol=output_scale), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data_f, output_data_f)

def test_TransposeConv2d_shape_4_4_1_1_int8_QuantizedPerChannelOneTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_1_1_int8_QuantizedPerChannelOneTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_scale, input_zp = 16.0 / 255, -128
    weight_scale, weight_zp = 9.0 / 127, 0
    bias_scale, bias_zp = input_scale * weight_scale, 0
    output_scale, output_zp = 2, -128

    input_name = "input"
    input_tensor_shape = [4, 4, 1, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = [input_scale]
    input_quant_info["zero_points"] = [input_zp]
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "INT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = [weight_scale]
    weight_quant_info["zero_points"] = [weight_zp]
    weight_quant_info["channel_dim"] = 3
    weight_quant_info["quant_type"] = "SYMMETRIC_PER_CHANNEL"
    weight_data = np.array([14, 28, 42, 56, 71, 85, 99, 113, 127]).reshape(weight_tensor_shape).astype(np.int8)
    assert timvx_engine.create_tensor(weight_name, "INT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, 
        np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [weight_tensor_shape[3]]
    bias_quant_info = {}
    bias_quant_info["scales"] = [bias_scale]
    bias_quant_info["zero_points"] = [bias_zp]
    bias_quant_info["channel_dim"] = 0
    bias_quant_info["quant_type"] = "SYMMETRIC_PER_CHANNEL"
    bias_data = np.array([0]).reshape(bias_tensor_shape).astype(np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, 
        np_data=bias_data), "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 4, 1, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = [output_scale]
    output_quant_info["zero_points"] = [output_zp]
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "INT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", "bias", ]
    op_outputs = ["output", ]
    padding = "SAME"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 4, 4]
    input_data_list = [1, 2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scale, input_zp, np.int8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [1, 1, 4, 4]
    golden_data_list = [28,  62,  82,  76,  98,  192, 238, 198,
                        206, 372, 416, 330, 262, 446, 486, 366]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    output_data_f = dequantize(output_data[0], output_scale, output_zp)
    assert np.allclose(golden_data_f, output_data_f, atol=output_scale), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data_f, output_data_f)

def test_TransposeConv2d_shape_2_2_1_1_int8_QuantizedPerChannelTwoTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_2_2_1_1_int8_QuantizedPerChannelTwoTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_scales, input_zps = [4.0 / 255], [-128]
    weight_scales, weight_zps = [17.0 / 127, 18.0 / 127], [0, 0]
    bias_scales, bias_zps = [input_scales[0] * weight_scales[0], input_scales[0] * weight_scales[1]], [0, 0]
    output_scales, output_zps = [1], [-128]

    input_name = "input"
    input_tensor_shape = [2, 2, 1, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = input_scales
    input_quant_info["zero_points"] = input_zps
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "INT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 2] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = weight_scales
    weight_quant_info["zero_points"] = weight_zps
    weight_quant_info["channel_dim"] = 3
    weight_quant_info["quant_type"] = "SYMMETRIC_PER_CHANNEL"
    weight_data = np.array([7,  22, 37, 52, 67, 82, 97, 112, 127,
                            14, 28, 42, 56, 71, 85, 99, 113, 127]).reshape(weight_tensor_shape).astype(np.int8)
    assert timvx_engine.create_tensor(weight_name, "INT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, 
        np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [weight_tensor_shape[3]]
    bias_quant_info = {}
    bias_quant_info["scales"] = bias_scales
    bias_quant_info["zero_points"] = bias_zps
    bias_quant_info["channel_dim"] = 0
    bias_quant_info["quant_type"] = "SYMMETRIC_PER_CHANNEL"
    bias_data = np.array([0, 0]).reshape(bias_tensor_shape).astype(np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, 
        np_data=bias_data), "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [5, 5, 2, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = output_scales
    output_quant_info["zero_points"] = output_zps
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "INT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", "bias", ]
    op_outputs = ["output", ]
    padding = "VALID"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [2, 2]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 2, 2]
    input_data_list = [1, 2, 3, 4]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scales[0], input_zps[0], np.int8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [1, 2, 5, 5]
    golden_data_list = [1,  3,  7,  6,  10, 7,   9,  25, 18, 22, 16, 24, 62, 42,  54, 21, 27,
                        61, 36, 44, 39, 45, 103, 60, 68, 2,  4,  10, 8,  12, 8,   10, 28, 20,
                        24, 20, 28, 72, 48, 60,  24, 30, 68, 40, 48, 42, 48, 110, 64, 72]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    output_data_f = dequantize(output_data[0], output_scales[0], output_zps[0])
    assert np.allclose(golden_data_f, output_data_f, atol=output_scales[0]), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data_f, output_data_f)

def test_TransposeConv2d_shape_4_4_1_1_int8_QuantizedBiasPerChannelTest():
    # create graph
    timvx_engine = Engine("test_TransposeConv2d_shape_4_4_1_1_int8_QuantizedBiasPerChannelTest")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_scales, input_zps = [16.0 / 255], [-128]
    weight_scales, weight_zps = [9.0 / 127], [0]
    bias_scales, bias_zps = [input_scales[0] * weight_scales[0]], [0]
    output_scales, output_zps = [2], [-128]

    input_name = "input"
    input_tensor_shape = [4, 4, 1, 1] # whcn
    input_quant_info = {}
    input_quant_info["scales"] = input_scales
    input_quant_info["zero_points"] = input_zps
    input_quant_info["channel_dim"] = 2
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "INT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whio
    weight_quant_info = {}
    weight_quant_info["scales"] = weight_scales
    weight_quant_info["zero_points"] = weight_zps
    weight_quant_info["channel_dim"] = 3
    weight_quant_info["quant_type"] = "SYMMETRIC_PER_CHANNEL"
    weight_data = np.array([14, 28, 42, 56, 71, 85, 99, 113, 127]).reshape(weight_tensor_shape).astype(np.int8)
    assert timvx_engine.create_tensor(weight_name, "INT8", "CONSTANT", weight_tensor_shape, quant_info=weight_quant_info, 
        np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [weight_tensor_shape[3]]
    bias_quant_info = {}
    bias_quant_info["scales"] = bias_scales
    bias_quant_info["zero_points"] = bias_zps
    bias_quant_info["channel_dim"] = 0
    bias_quant_info["quant_type"] = "SYMMETRIC_PER_CHANNEL"
    bias_data = np.array([224]).reshape(bias_tensor_shape).astype(np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, 
        np_data=bias_data), "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [4, 4, 1, 1]  # whcn
    output_quant_info = {}
    output_quant_info["scales"] = output_scales
    output_quant_info["zero_points"] = output_zps
    output_quant_info["channel_dim"] = 2
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "INT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight", "bias", ]
    op_outputs = ["output", ]
    padding = "SAME"
    kernel_size = [weight_tensor_shape[1], weight_tensor_shape[0]]
    stride = [1, 1]
    output_padding = [0, 0]
    pad_left_inter = int((weight_tensor_shape[0] + stride[0] * (input_tensor_shape[0] - 1) - output_tensor_shape[1]) / 2)
    pad_left = pad_left_inter if pad_left_inter > 0 else 0
    pad_right = pad_left
    pad_top_inter = int((weight_tensor_shape[1] + stride[1] * (input_tensor_shape[1] - 1) - output_tensor_shape[0]) / 2)
    pad_top = pad_top_inter if pad_top_inter > 0 else 0
    pad_bottom = pad_top
    pad = [pad_left, pad_right, pad_top, pad_bottom]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=weight_tensor_shape[3], 
        pad_type=padding, ksize=kernel_size, stride=stride, output_padding=output_padding, 
        pad=pad, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 4, 4]
    input_data_list = [1, 2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16]
    input_data_f = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_data = quantize(input_data_f, input_scales[0], input_zps[0], np.int8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict, pass_through=True, want_float=False)

    # compare gloden data with output data
    output_np_shape = [1, 1, 4, 4]
    golden_data_list = [30,  62,  84,  76,  100, 194, 238, 200,
                        208, 372, 418, 330, 264, 446, 486, 366]
    golden_data_f = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    output_data_f = dequantize(output_data[0], output_scales[0], output_zps[0])
    assert np.allclose(golden_data_f, output_data_f, atol=output_scales[0]), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data_f, output_data_f)

test_func_map = {}
test_func_map["TransposeConv2d_shape_4_4_1_1_float32_SimpleTest"] = test_TransposeConv2d_shape_4_4_1_1_float32_SimpleTest
test_func_map["TransposeConv2d_shape_4_4_2_1_float32_SameTest"] = test_TransposeConv2d_shape_4_4_2_1_float32_SameTest
test_func_map["TransposeConv2d_shape_4_4_2_1_float32_ValidTest"] = test_TransposeConv2d_shape_4_4_2_1_float32_ValidTest
test_func_map["TransposeConv2d_shape_2_2_1_1_float32_StrideTest"] = test_TransposeConv2d_shape_2_2_1_1_float32_StrideTest
test_func_map["TransposeConv2d_shape_2_2_1_1_float32_ChannelTest"] = test_TransposeConv2d_shape_2_2_1_1_float32_ChannelTest
test_func_map["TransposeConv2d_shape_2_1_1_1_float32_AccuracyTest"] = test_TransposeConv2d_shape_2_1_1_1_float32_AccuracyTest
test_func_map["TransposeConv2d_shape_2_2_1_1_float32_BiasChannelTest"] = test_TransposeConv2d_shape_2_2_1_1_float32_BiasChannelTest
test_func_map["TransposeConv2d_shape_4_4_1_1_uint8_QuantizedTest"] = test_TransposeConv2d_shape_4_4_1_1_uint8_QuantizedTest
test_func_map["TransposeConv2d_shape_4_4_2_1_uint8_QuantizedTwoFiltersTest"] = test_TransposeConv2d_shape_4_4_2_1_uint8_QuantizedTwoFiltersTest
test_func_map["TransposeConv2d_shape_4_4_2_1_uint8_QuantizedValidTest"] = test_TransposeConv2d_shape_4_4_2_1_uint8_QuantizedValidTest
test_func_map["TransposeConv2d_shape_4_4_1_1_uint8_QuantizedBiasTest"] = test_TransposeConv2d_shape_4_4_1_1_uint8_QuantizedBiasTest
test_func_map["TransposeConv2d_shape_4_4_1_1_int8_QuantizedPerChannelOneTest"] = test_TransposeConv2d_shape_4_4_1_1_int8_QuantizedPerChannelOneTest
test_func_map["TransposeConv2d_shape_2_2_1_1_int8_QuantizedPerChannelTwoTest"] = test_TransposeConv2d_shape_2_2_1_1_int8_QuantizedPerChannelTwoTest
test_func_map["TransposeConv2d_shape_4_4_1_1_int8_QuantizedBiasPerChannelTest"] = test_TransposeConv2d_shape_4_4_1_1_int8_QuantizedBiasPerChannelTest

def test_transpose_conv_op():
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
    test_transpose_conv_op()