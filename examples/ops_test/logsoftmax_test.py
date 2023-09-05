# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_LogSoftmax_shape_6_1_float_axis_0():
    # create graph
    timvx_engine = Engine("test_LogSoftmax_shape_6_1_float_axis_0")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [6, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [6, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "logsoftmax"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructLogSoftmaxOpConfig(op_name=op_name, axis=0, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6]
    input_data_list = [2, 3, 4, 5, 6, 7]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 6]
    golden_data_list = [-5.4562, -4.4562, -3.4562, -2.4562, -1.4562, -0.4562,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_LogSoftmax_shape_3_6_1_float_axis_1():
    # create graph
    timvx_engine = Engine("test_LogSoftmax_shape_3_6_1_float_axis_1")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 6, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [3, 6, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "logsoftmax"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructLogSoftmaxOpConfig(op_name=op_name, axis=1, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6, 3]
    input_data_list = [-2.0000,  0.0000,  2.0000,
                       -3.0000,  0.0000,  3.0000,
                       -4.0000,  0.0000,  4.0000,
                       -5.0000,  0.0000,  5.0000,
                       -6.0000,  0.0000,  6.0000,
                       -7.0000,  0.0000,  7.0000,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 6, 3]
    golden_data_list = [-0.4561933, -1.7917595, -5.4561934,
                        -1.4561933, -1.7917595, -4.4561934,
                        -2.4561934, -1.7917595, -3.4561934,
                        -3.4561934, -1.7917595, -2.4561934,
                        -4.4561934, -1.7917595, -1.4561933,
                        -5.4561934, -1.7917595, -0.4561933,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_LogSoftmax_shape_3_6_1_uint8_axis_1():
    # create graph
    timvx_engine = Engine("test_LogSoftmax_shape_3_6_1_uint8_axis_1")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 6, 1]
    input_scale = 1
    input_zp = 2
    input_quant_info = {}
    input_quant_info["scale"] = input_scale
    input_quant_info["zero_point"] = input_zp
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [3, 6, 1]
    output_scale = 1.7917595
    output_zp = 2
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "logsoftmax"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructLogSoftmaxOpConfig(op_name=op_name, axis=1, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6, 3]
    input_data_list = [0,  2,  4,
                       0,  2,  4,
                       0,  2,  4,
                       0,  2,  4,
                       0,  2,  4,
                       0,  2,  4,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 6, 3]
    golden_data_list = [1,  1,  1,
                        1,  1,  1,
                        1,  1,  1,
                        1,  1,  1,
                        1,  1,  1,
                        1,  1,  1,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["LogSoftmax_shape_6_1_float_axis_0"] = test_LogSoftmax_shape_6_1_float_axis_0
test_func_map["LogSoftmax_shape_3_6_1_float_axis_1"] = test_LogSoftmax_shape_3_6_1_float_axis_1
test_func_map["LogSoftmax_shape_3_6_1_uint8_axis_1"] = test_LogSoftmax_shape_3_6_1_uint8_axis_1

def test_logsoftmax_op():
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
    test_logsoftmax_op()