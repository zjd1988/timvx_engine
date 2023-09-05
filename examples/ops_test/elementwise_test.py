# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_FloorDiv_shape_1_fp32():
    # create graph
    timvx_engine = Engine("test_FloorDiv_shape_1_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name1 = "input1"
    assert timvx_engine.create_tensor(input_name1, "FLOAT32", "INPUT", [1,]), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert timvx_engine.create_tensor(input_name2, "FLOAT32", "INPUT", [1,]), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", [1,]), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "FloorDiv"
    op_inputs = [input_name1, input_name2, ]
    op_outputs = [output_name, ]
    op_info = ConstructEltwiseOpConfig(op_name=op_name, eltwise_type="FloorDiv", 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    input_data1 = np.array([1,]).astype(np.float32)
    input_data2 = np.array([0,]).astype(np.float32)
    input_dict = {}
    input_dict[input_name1] = input_data1
    input_dict[input_name2] = input_data2
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    golden_data = np.array([float('inf'),]).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_FloorDiv_shape_5_1_broadcast_float32():
    # create graph
    timvx_engine = Engine("test_FloorDiv_shape_5_1_broadcast_float32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name1 = "x"
    input_tensor_shape1 = [5, 1]
    assert timvx_engine.create_tensor(input_name1, "FLOAT32", "INPUT", input_tensor_shape1), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "y"
    input_tensor_shape2 = [1,]
    assert timvx_engine.create_tensor(input_name2, "FLOAT32", "INPUT", input_tensor_shape2), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    output_tensor_shape = [5, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "FloorDiv"
    op_inputs = [input_name1, input_name2, ]
    op_outputs = [output_name, ]
    op_info = ConstructEltwiseOpConfig(op_name=op_name, eltwise_type="FloorDiv", 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    input_np_shape1 = [1, 5]
    input_np_shape2 = [1,]
    x = np.array([1, 3, -2, 0, 99]).reshape(input_np_shape1).astype(np.float32)
    y = np.array([2,]).reshape(input_np_shape2).astype(np.float32)
    input_dict = {}
    input_dict[input_name1] = x
    input_dict[input_name2] = y
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 5]
    golden_data = np.array([0, 1, -1, 0, 49]).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_FloorDiv_shape_5_1_broadcast_uint8():
    # create graph
    timvx_engine = Engine("test_FloorDiv_shape_5_1_broadcast_uint8")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name1 = "x"
    input_tensor_shape1 = [1,]
    input_quant_info1 = {}
    input_quant_info1["scale"] = 1
    input_quant_info1["zero_point"] = 0
    input_quant_info1["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name1, "UINT8", "INPUT", input_tensor_shape1, quant_info=input_quant_info1), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "y"
    input_tensor_shape2 = [5, 1]
    input_quant_info2 = {}
    input_quant_info2["scale"] = 1
    input_quant_info2["zero_point"] = 0
    input_quant_info2["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name2, "UINT8", "INPUT", input_tensor_shape2, quant_info=input_quant_info2), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    output_tensor_shape = [5, 1]
    output_quant_info = {}
    output_quant_info["scale"] = 0.5
    output_quant_info["zero_point"] = 0
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "FloorDiv"
    op_inputs = [input_name1, input_name2, ]
    op_outputs = [output_name, ]
    op_info = ConstructEltwiseOpConfig(op_name=op_name, eltwise_type="FloorDiv", 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    input_np_shape1 = [1,]
    input_np_shape2 = [1, 5]
    x = np.array([255, ]).reshape(input_np_shape1).astype(np.uint8)
    y = np.array([1, 3, 2, 0, 255]).reshape(input_np_shape2).astype(np.uint8)
    input_dict = {}
    input_dict[input_name1] = x
    input_dict[input_name2] = y
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 5]
    golden_data = np.array([255, 170, 254, 255, 2]).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["FloorDiv_shape_1_fp32"] = test_FloorDiv_shape_1_fp32
test_func_map["FloorDiv_shape_5_1_broadcast_float32"] = test_FloorDiv_shape_5_1_broadcast_float32
test_func_map["FloorDiv_shape_5_1_broadcast_uint8"] = test_FloorDiv_shape_5_1_broadcast_uint8

def test_elementwise_op():
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
    test_elementwise_op()