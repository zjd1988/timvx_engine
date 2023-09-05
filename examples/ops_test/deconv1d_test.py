# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_DeConv1d_no_bias_layout_whcn_depthwise_shape_3_2_1():
    # create graph
    timvx_engine = Engine("test_DeConv1d_no_bias_layout_whcn_depthwise_shape_3_2_1")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 2, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 2, 1] # whc1 same as depthwise convolution
    weight_data = np.array([9.0, 0.0, 1.0,
                            3.0, 0.0, 0.0]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [5, 2, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv1d"
    op_inputs = ["input", "weight"]
    op_outputs = ["output", ]
    op_info = ConstructDeConv1dOpConfig(op_name=op_name, oc_count=2, pad_type="SAME", ksize=3, 
        stride=1, output_padding=1, pad=[0, 0], group=2, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 3]
    input_data_list = [3.0, 9.0, 3.0,
                       7.0, 5.0, 9.0,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 5]
    golden_data_list = [27.0, 81.0, 30.0, 9.0, 3.0,
                        21.0, 15.0, 27.0, 0.0, 0.0]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def  test_DeConv1d_layout_whcn_shape_3_1_1():
    # create graph
    timvx_engine = Engine("test_DeConv1d_layout_whcn_shape_3_1_1")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 1, 1]
    input_scale = 1.0
    input_zp = 0
    input_quant_info = {}
    input_quant_info["scale"] = input_scale
    input_quant_info["zero_point"] = input_zp
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, \
        quant_info=input_quant_info), "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 1, 1]
    weight_scale = 1.0
    weight_zp = 0
    weight_quant_info = {}
    weight_quant_info["scale"] = weight_scale
    weight_quant_info["zero_point"] = weight_zp
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data = np.array([9, 0, 1]).reshape(weight_tensor_shape).astype(np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, \
        quant_info=weight_quant_info, np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [1,]
    bias_scale = 1.0
    bias_zp = 0
    bias_quant_info = {}
    bias_quant_info["scale"] = bias_scale
    bias_quant_info["zero_point"] = bias_zp
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    bias_data = np.array([-5, ]).reshape(bias_tensor_shape).astype(np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, \
        quant_info=bias_quant_info, np_data=bias_data), "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [5, 1, 1]
    output_scale = 1.0
    output_zp = 2
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, \
        quant_info=output_quant_info), "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv1d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructDeConv1dOpConfig(op_name=op_name, oc_count=1, pad_type="SAME", ksize=3, 
        stride=1, output_padding=1, pad=[0, 0], group=1, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 3]
    input_data_list = [3, 9, 3]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 5]
    golden_data_list = [24, 78, 27, 6, 0,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["DeConv1d_no_bias_layout_whcn_depthwise_shape_3_2_1"] = test_DeConv1d_no_bias_layout_whcn_depthwise_shape_3_2_1
test_func_map["DeConv1d_layout_whcn_shape_3_1_1"] = test_DeConv1d_layout_whcn_shape_3_1_1

def test_deconv1d_op():
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
    test_deconv1d_op()