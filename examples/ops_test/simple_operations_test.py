# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Floor_shape_5_1_fp32():
    # create graph
    timvx_engine = Engine("test_Floor_shape_5_1_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [5, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [5, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "simple_operations"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructSimpleOperationsOpConfig(op_name=op_name, simple_type='Floor', 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 5]
    input_data_list = [-2.5, -0.1, 0, 0.55, float('inf')]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 5]
    golden_data_list = [-3, -1, 0, 0, float('inf')]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Cast_shape_5_1_fp32_to_int32():
    # create graph
    timvx_engine = Engine("test_Cast_shape_5_1_fp32_to_int32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [5, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [5, 1]
    assert timvx_engine.create_tensor(output_name, "INT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "simple_operations"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructSimpleOperationsOpConfig(op_name=op_name, simple_type='Cast', 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 5]
    input_data_list = [-2.5, -0.1, 0, 0.55, float('inf')]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 5]
    golden_data_list = [-2, 0, 0, 0, np.iinfo(np.int32).max]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.int32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["Floor_shape_5_1_fp32"] = test_Floor_shape_5_1_fp32
test_func_map["Cast_shape_5_1_fp32_to_int32"] = test_Cast_shape_5_1_fp32_to_int32

def test_simple_operations_op():
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
    test_simple_operations_op()