# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Linear_shape_5_1_fp32():
    # create graph
    timvx_engine = Engine("test_Linear_shape_5_1_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    tensor_shape = [5, 1]
    input_name = "input"
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "linear"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_parameter = {}
    op_parameter["a"] = 1
    op_parameter["b"] = 2
    op_info = ConstructActivationOpConfig(op_name=op_name, activation_type="Linear", parameter=op_parameter, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    np_shape = [1, 5]
    input_data = np.array([-2.5, -0.1, 0, 0.55, float('inf')]).reshape(np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    golden_data = np.array([-0.5, 1.9, 2.0, 2.55, float('inf')]).reshape(np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Linear_shape_5_1_fp32_omit_b():
    # create graph
    timvx_engine = Engine("test_Linear_shape_5_1_fp32_omit_b")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    tensor_shape = [5, 1]
    input_name = "input"
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "linear"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_parameter = {}
    op_parameter["a"] = 2
    op_info = ConstructActivationOpConfig(op_name=op_name, activation_type="Linear", parameter=op_parameter, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."
    
    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    np_shape = [1, 5]
    input_data = np.array([-2.5, -0.1, 0, 0.55, float('inf')]).reshape(np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    golden_data = np.array([-5.0, -0.2, 0, 1.1, float('inf')]).reshape(np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["Linear_shape_5_1_fp32"] = test_Linear_shape_5_1_fp32
test_func_map["Linear_shape_5_1_fp32_omit_b"] = test_Linear_shape_5_1_fp32_omit_b

def test_activations_op():
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
    test_activations_op()

