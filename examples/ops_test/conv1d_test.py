# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Conv1d_shape_3_6_1_float_ksize_1_stride_1_weights_3_no_bias_whcn():
    # create graph
    timvx_engine = Engine("test_Conv1d_shape_3_6_1_float_ksize_1_stride_1_weights_3_no_bias_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 6, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    param_name = "param"
    param_tensor_shape = [1, 6, 3]
    param_data = np.array([-3,   -2, -1.5, 1.5, 2, 3,
        -2.5, -2, -1.5, 1.5, 2, 2.5,
        -2.5, -2, 0,    0,   2, 2.5,]).reshape(param_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(param_name, "FLOAT32", "INPUT", param_tensor_shape, np_data=param_data), \
        "construct tensor {} fail!".format(param_name)

    output_name = "output"
    output_tensor_shape = [3, 3, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv1d"
    op_inputs = ["input", "param"]
    op_outputs = ["output", ]
    op_info = ConstructConv1dOpConfig(op_name=op_name, weights=3, padding="VALID", ksize=1, 
        stride=1, dilation=1, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."
    
    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6, 3]
    input_data_list = [-1,    0,   1,
        -1.5,  0.5, 1.5,
        -2,   -0.5, 2,
        -2.5,  0,   2.5,
        -3,    0.5, 3,
        -3.5,  0.5, 3.5]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 3, 3]
    golden_data_list = [-11.25, 2.25, 11.25,
        -10,    2,    10,
        -9.25,  1.25, 9.25,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv1d_shape_6_2_1_uint8_ksize_6_stride_1_weights_2_whcn():
    # create graph
    timvx_engine = Engine("test_Conv1d_shape_3_6_1_float_ksize_1_stride_1_weights_3_no_bias_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [6, 2, 1]
    input_quant_info = {}
    input_quant_info["scale"] = 0.25
    input_quant_info["zero_point"] = 6
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    param_name = "param"
    param_tensor_shape = [6, 2, 2]
    param_data = np.array([12, 14,
        16, 28,
        30, 32,
         8, 10,
        12, 32,
        34, 36,
         4,  6,
         8, 36,
        38, 40,
         0,  2,
         4, 40,
        42, 44]).reshape(param_tensor_shape).astype(np.uint8)
    param_quant_info = {}
    param_quant_info["scale"] = 0.25
    param_quant_info["zero_point"] = 22
    param_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(param_name, "UINT8", "CONSTANT", param_tensor_shape, quant_info=param_quant_info, np_data=param_data), \
        "construct tensor {} fail!".format(param_name)

    bias_name = "bias"
    bias_tensor_shape = [2,]
    bias_data = np.array([-20, 100]).reshape(bias_tensor_shape).astype(np.int32)
    bias_quant_info = {}
    bias_quant_info["scale"] = 0.0625
    bias_quant_info["zero_point"] = 0
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, quant_info=bias_quant_info, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [1, 2, 1]
    output_quant_info = {}
    output_quant_info["scale"] = 0.25
    output_quant_info["zero_point"] = 0
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv1d"
    op_inputs = ["input", "param", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructConv1dOpConfig(op_name=op_name, weights=2, padding="VALID", ksize=6, 
        stride=1, dilation=1, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."
    
    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 6]
    input_data_list = [4,  5,  6,  6,  7,  8,
        0,  2,  4,  8, 10, 12,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 1]
    golden_data_list = [85, 175]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Conv1d_shape_6_2_1_uint8_ksize_3_stride_1_pad_1_weights_2_no_bias_whcn():
    # create graph
    timvx_engine = Engine("test_Conv1d_shape_6_2_1_uint8_ksize_3_stride_1_pad_1_weights_2_no_bias_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [6, 2, 1]
    input_quant_info = {}
    input_quant_info["scale"] = 0.25
    input_quant_info["zero_point"] = 6
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    param_name = "param"
    param_tensor_shape = [3, 2, 2]
    param_data = np.array([12, 14, 16,
         8, 10, 12,
         4,  6,  8,
         0,  2,  4,]).reshape(param_tensor_shape).astype(np.uint8)
    param_quant_info = {}
    param_quant_info["scale"] = 0.25
    param_quant_info["zero_point"] = 22
    param_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(param_name, "UINT8", "CONSTANT", param_tensor_shape, quant_info=param_quant_info, np_data=param_data), \
        "construct tensor {} fail!".format(param_name)

    output_name = "output"
    output_tensor_shape = [3, 2, 1]
    output_quant_info = {}
    output_quant_info["scale"] = 0.25
    output_quant_info["zero_point"] = 69
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "conv1d"
    op_inputs = ["input", "param"]
    op_outputs = ["output", ]
    op_info = ConstructConv1dOpConfig(op_name=op_name, weights=2, padding="AUTO", ksize=3, 
        stride=2, dilation=1, pad=[0, 1], op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."
    
    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 6]
    input_data_list = [4,  4,  6,  6,  8,  8,
        0,  2,  4,  8, 10, 12,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 3]
    golden_data_list = [116, 57, 28,
        148, 45,  0,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])


test_func_map = {}
test_func_map["Conv1d_shape_3_6_1_float_ksize_1_stride_1_weights_3_no_bias_whcn"] = test_Conv1d_shape_3_6_1_float_ksize_1_stride_1_weights_3_no_bias_whcn
test_func_map["Conv1d_shape_6_2_1_uint8_ksize_6_stride_1_weights_2_whcn"] = test_Conv1d_shape_6_2_1_uint8_ksize_6_stride_1_weights_2_whcn
test_func_map["Conv1d_shape_6_2_1_uint8_ksize_3_stride_1_pad_1_weights_2_no_bias_whcn"] = test_Conv1d_shape_6_2_1_uint8_ksize_3_stride_1_pad_1_weights_2_no_bias_whcn

def test_conv1d_op():
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
    test_conv1d_op()
