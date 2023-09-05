# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_InstanceNorm_shape_3_6_1_float():
    # create graph
    timvx_engine = Engine("test_InstanceNorm_shape_3_6_1_float")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 6, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    gamma_name = "gamma"
    gamma_tensor_shape = [6,]
    gamma_data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(gamma_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(gamma_name, "FLOAT32", "CONSTANT", gamma_tensor_shape, np_data=gamma_data), \
        "construct tensor {} fail!".format(gamma_name)

    beta_name = "beta"
    beta_tensor_shape = [6,]
    beta_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(beta_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(beta_name, "FLOAT32", "CONSTANT", beta_tensor_shape, np_data=beta_data), \
        "construct tensor {} fail!".format(beta_name)

    output_name = "output"
    output_tensor_shape = [3, 6, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "instancenormalization"
    op_inputs = ["input", "beta", "gamma"]
    op_outputs = ["output", ]
    op_info = ConstructInstanceNormalizationOpConfig(op_name=op_name, eps=2e-5, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6, 3]
    input_data_list = [-2, 0, 2,
                       -3, 0, 3,
                       -4, 0, 4,
                       -5, 0, 5,
                       -6, 0, 6,
                       -7, 0, 7,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 6, 3]
    golden_data_list = [-1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_InstanceNorm_shape_3_3_6_1_float():
    # create graph
    timvx_engine = Engine("test_InstanceNorm_shape_3_3_6_1_float")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [2, 3, 6, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    gamma_name = "gamma"
    gamma_tensor_shape = [6,]
    gamma_data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(gamma_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(gamma_name, "FLOAT32", "CONSTANT", gamma_tensor_shape, np_data=gamma_data), \
        "construct tensor {} fail!".format(gamma_name)

    beta_name = "beta"
    beta_tensor_shape = [6,]
    beta_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(beta_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(beta_name, "FLOAT32", "CONSTANT", beta_tensor_shape, np_data=beta_data), \
        "construct tensor {} fail!".format(beta_name)

    output_name = "output"
    output_tensor_shape = [2, 3, 6, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "instancenormalization"
    op_inputs = ["input", "beta", "gamma"]
    op_outputs = ["output", ]
    op_info = ConstructInstanceNormalizationOpConfig(op_name=op_name, eps=2e-5, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6, 3, 2]
    input_data_list = [-2, 0, 2, -2, 0, 2,
                       -3, 0, 3, -3, 0, 3,
                       -4, 0, 4, -4, 0, 4,
                       -5, 0, 5, -5, 0, 5,
                       -6, 0, 6, -6, 0, 6,
                       -7, 0, 7, -7, 0, 7,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 6, 3, 2]
    golden_data_list = [-1.22474, 0, 1.22474, -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474, -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474, -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474, -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474, -1.22474, 0, 1.22474,
                        -1.22474, 0, 1.22474, -1.22474, 0, 1.22474,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])


test_func_map = {}
test_func_map["InstanceNorm_shape_3_6_1_float"] = test_InstanceNorm_shape_3_6_1_float
test_func_map["InstanceNorm_shape_3_3_6_1_float"] = test_InstanceNorm_shape_3_3_6_1_float

def test_instancenormalization_op():
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
    test_instancenormalization_op()