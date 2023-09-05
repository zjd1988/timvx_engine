# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_DeConv2d_shape_3_3_2_1_float_depthwise():
    # create graph
    timvx_engine = Engine("test_DeConv2d_shape_3_3_2_1_float_depthwise")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 3, 2, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 2, 1] # whc1 same as depthwise convolution
    weight_data = np.array([9.0, 0.0, 3.0,
                            0.0, 0.0, 0.0,
                            1.0, 0.0, 2.0,
                            3.0, 0.0, 7.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 8.0,]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [5, 5, 2, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight"]
    op_outputs = ["output", ]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=2, pad_type="SAME", ksize=[3, 3], 
        stride=[1, 1], output_padding=[1, 1], pad=[0, 0, 0, 0], group=2, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 3, 3]
    input_data_list = [3.0, 8.0, 1.0,
                       9.0, 5.0, 7.0,
                       3.0, 2.0, 3.0,
                       7.0, 9.0, 1.0,
                       5.0, 2.0, 3.0,
                       9.0, 0.0, 2.0]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 5, 5]
    golden_data_list = [27.0, 72.0, 18.0, 24.0, 3.0,
                        81.0, 45.0, 90.0, 15.0, 21.0,
                        30.0, 26.0, 43.0, 22.0, 11.0,
                        9.0, 5.0, 25.0, 10.0, 14.0,
                        3.0, 2.0, 9.0, 4.0, 6.0,
                        21.0, 27.0, 52.0, 63.0, 7.0,
                        15.0, 6.0, 44.0, 14.0, 21.0,
                        27.0, 0.0, 125.0, 72.0, 22.0,
                        0.0, 0.0, 40.0, 16.0, 24.0,
                        0.0, 0.0, 72.0, 0.0, 16.0]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_DeConv2d_shape_3_3_1_1_float():
    # create graph
    timvx_engine = Engine("test_DeConv2d_shape_3_3_2_1_float_depthwise")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 3, 1, 1] # whcn
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3, 3, 1, 1] # whc1 same as depthwise convolution
    weight_data = np.array([9.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [5, 5, 1, 1] # whcn
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "deconv2d"
    op_inputs = ["input", "weight"]
    op_outputs = ["output", ]
    op_info = ConstructDeConv2dOpConfig(op_name=op_name, oc_count=1, pad_type="SAME", ksize=[3, 3], 
        stride=[1, 1], output_padding=[1, 1], pad=[0, 0, 0, 0], group=1, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 3, 3]
    input_data_list = [3.0, 8.0, 1.0, 9.0, 5.0, 7.0, 3.0, 2.0, 3.0]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 5, 5]
    golden_data_list = [27.0, 72.0, 18.0, 24.0, 3.0,  81.0, 45.0, 90.0, 15.0,
                        21.0, 30.0, 26.0, 43.0, 22.0, 11.0, 9.0,  5.0,  25.0,
                        10.0, 14.0, 3.0,  2.0,  9.0,  4.0,  6.0]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["DeConv2d_shape_3_3_2_1_float_depthwise"] = test_DeConv2d_shape_3_3_2_1_float_depthwise
test_func_map["DeConv2d_shape_3_3_1_1_float"] = test_DeConv2d_shape_3_3_1_1_float

def test_deconv2d_op():
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
    test_deconv2d_op()