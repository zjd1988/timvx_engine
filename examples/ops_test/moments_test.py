# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Moments_shape_6_3_1_float_axes_0_1():
    # create graph
    timvx_engine = Engine("test_Moments_shape_6_3_1_float_axes_0_1")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [6, 3, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_mean_name = "mean"
    output_mean_shape = [1,]
    assert timvx_engine.create_tensor(output_mean_name, "FLOAT32", "OUTPUT", output_mean_shape), \
        "construct tensor {} fail!".format(output_mean_name)

    output_variance_name = "variance"
    output_variance_shape = [1,]
    assert timvx_engine.create_tensor(output_variance_name, "FLOAT32", "OUTPUT", output_variance_shape), \
        "construct tensor {} fail!".format(output_variance_name)

    # construct operations
    op_name = "moments"
    op_inputs = ["input", ]
    op_outputs = ["mean", "variance", ]
    op_info = ConstructMomentsOpConfig(op_name=op_name, axes=[0, 1], op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, 3, 6]
    input_data_list = [-2, 0, 2,
                       -3, 0, 3,
                       -4, 0, 4,
                       -5, 0, 5,
                       -6, 0, 6,
                       -7, 0, 7]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict["input"] = input_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, ]
    golden_mean_list = [0,]
    golden_mean_data = np.array(golden_mean_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_mean_data, output_data[0], atol=1.e-6), \
        "check gloden mean data with output data not equal!\n gloden:{}\n output:{}".format(golden_mean_data, output_data[0])

    golden_variance_list = [15.444444,]
    golden_variance_data = np.array(golden_variance_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_variance_data, output_data[1], atol=1.e-6), \
        "check gloden variance data with output data not equal!\n gloden:{}\n output:{}".format(golden_variance_data, output_data[1])

def test_Moments_shape_3_6_1_float_axes_1_keepdims():
    # create graph
    timvx_engine = Engine("test_Moments_shape_3_6_1_float_axes_1_keepdims")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 6, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_mean_name = "mean"
    output_mean_shape = [3, 1, 1,]
    assert timvx_engine.create_tensor(output_mean_name, "FLOAT32", "OUTPUT", output_mean_shape), \
        "construct tensor {} fail!".format(output_mean_name)

    output_variance_name = "variance"
    output_variance_shape = [3, 1, 1,]
    assert timvx_engine.create_tensor(output_variance_name, "FLOAT32", "OUTPUT", output_variance_shape), \
        "construct tensor {} fail!".format(output_variance_name)

    # construct operations
    op_name = "moments"
    op_inputs = ["input", ]
    op_outputs = ["mean", "variance", ]
    op_info = ConstructMomentsOpConfig(op_name=op_name, axes=[1, ], keep_dims=True, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, 6, 3]
    input_data_list = [ -2, 0, 2,
                        -3, 0, 3,
                        -4, 0, 4,
                        -5, 0, 5,
                        -6, 0, 6,
                        -7, 0, 7 ]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict["input"] = input_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 3]
    golden_mean_list = [-4.5, 0, 4.5,]
    golden_mean_data = np.array(golden_mean_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_mean_data, output_data[0], atol=1.e-6), \
        "check gloden mean data with output data not equal!\n gloden:{}\n output:{}".format(golden_mean_data, output_data[0])

    golden_variance_list = [2.916666, 0, 2.916666,]
    golden_variance_data = np.array(golden_variance_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_variance_data, output_data[1], atol=1.e-6), \
        "check gloden variance data with output data not equal!\n gloden:{}\n output:{}".format(golden_variance_data, output_data[1])

test_func_map = {}
test_func_map["Moments_shape_6_3_1_float_axes_0_1"] = test_Moments_shape_6_3_1_float_axes_0_1
test_func_map["Moments_shape_3_6_1_float_axes_1_keepdims"] = test_Moments_shape_3_6_1_float_axes_1_keepdims

def test_moments_op():
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
    test_moments_op()