# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Matmul_shape_2_6_shape_6_2_float():
    # create graph
    timvx_engine = Engine("test_Matmul_shape_2_6_shape_6_2_float")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    a_name = "a"
    a_tensor_shape = [6, 2]
    assert timvx_engine.create_tensor(a_name, "FLOAT32", "INPUT", a_tensor_shape), \
        "construct tensor {} fail!".format(a_name)

    b_name = "b"
    b_tensor_shape = [2, 6]
    assert timvx_engine.create_tensor(b_name, "FLOAT32", "INPUT", b_tensor_shape), \
        "construct tensor {} fail!".format(b_name)

    output_name = "output"
    output_tensor_shape = [2, 2]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "matmul"
    op_inputs = ["a", "b", ]
    op_outputs = ["output", ]
    op_info = ConstructMatmulOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    a_np_shape = [2, 6]
    a_data_list = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
    a_data = np.array(a_data_list).reshape(a_np_shape).astype(np.float32)
    input_dict["a"] = a_data

    b_np_shape = [6, 2]
    b_data_list = [6, 5, 4, 3, 2, 1, -6, -5, -4, -3, -2, -1]
    b_data = np.array(b_data_list).reshape(b_np_shape).astype(np.float32)
    input_dict["b"] = b_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 2]
    golden_data_list = [-36, -27, 36, 27,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Matmul_shape_2_3_2_shape_2_3_2_float_transpose_b():
    # create graph
    timvx_engine = Engine("test_Matmul_shape_2_3_2_shape_2_3_2_float_transpose_b")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    a_name = "a"
    a_tensor_shape = [2, 3, 2]
    assert timvx_engine.create_tensor(a_name, "FLOAT32", "INPUT", a_tensor_shape), \
        "construct tensor {} fail!".format(a_name)

    b_name = "b"
    b_tensor_shape = [2, 3, 2]
    assert timvx_engine.create_tensor(b_name, "FLOAT32", "INPUT", b_tensor_shape), \
        "construct tensor {} fail!".format(b_name)

    output_name = "output"
    output_tensor_shape = [3, 3, 2]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "matmul"
    op_inputs = ["a", "b", ]
    op_outputs = ["output", ]
    op_info = ConstructMatmulOpConfig(op_name=op_name, transpose_a=False, transpose_b=True, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    a_np_shape = [2, 3, 2]
    a_data_list = [1, 2, 3, 4, 5, 6, -1, -2,-3, -4, -5, -6]
    a_data = np.array(a_data_list).reshape(a_np_shape).astype(np.float32)
    input_dict["a"] = a_data

    b_np_shape = [2, 3, 2]
    b_data_list = [6, 5, 4, 3, 2, 1, -6, -5, -4, -3, -2, -1]
    b_data = np.array(b_data_list).reshape(b_np_shape).astype(np.float32)
    input_dict["b"] = b_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 3, 3]
    golden_data_list = [16, 10,  4,
                        38, 24, 10,
                        60, 38, 16,
                        16, 10,  4,
                        38, 24, 10,
                        60, 38, 16,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Matmul_shape_2_3_2_shape_2_3_2_uint8_transpose_a():
    # create graph
    timvx_engine = Engine("test_Matmul_shape_2_3_2_shape_2_3_2_uint8_transpose_a")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    a_name = "a"
    a_tensor_shape = [2, 3, 2]
    a_scale = 1
    a_zp = 6
    a_quant_info = {}
    a_quant_info["scale"] = a_scale
    a_quant_info["zero_point"] = a_zp
    a_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(a_name, "UINT8", "INPUT", a_tensor_shape, quant_info=a_quant_info), \
        "construct tensor {} fail!".format(a_name)

    b_name = "b"
    b_tensor_shape = [2, 3, 2]
    b_scale = 1
    b_zp = 6
    b_quant_info = {}
    b_quant_info["scale"] = b_scale
    b_quant_info["zero_point"] = b_zp
    b_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(b_name, "UINT8", "INPUT", b_tensor_shape, quant_info=b_quant_info), \
        "construct tensor {} fail!".format(b_name)

    output_name = "output"
    output_tensor_shape = [2, 2, 2]
    output_scale = 1
    output_zp = 0
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "matmul"
    op_inputs = ["a", "b", ]
    op_outputs = ["output", ]
    op_info = ConstructMatmulOpConfig(op_name=op_name, transpose_a=True,  
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    a_np_shape = [2, 3, 2]
    a_data_list = [7, 8, 9, 10, 11, 12, 5, 4, 3, 2, 1, 0,]
    a_data = np.array(a_data_list).reshape(a_np_shape).astype(np.uint8)
    input_dict["a"] = a_data

    b_np_shape = [2, 3, 2]
    b_data_list = [12, 11, 10, 9, 8, 7, 0, 1, 2, 3, 4, 5,]
    b_data = np.array(b_data_list).reshape(b_np_shape).astype(np.uint8)
    input_dict["b"] = b_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [2, 2, 2]
    golden_data_list = [28, 19,
                        40, 28,
                        28, 19,
                        40, 28,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["Matmul_shape_2_6_shape_6_2_float"] = test_Matmul_shape_2_6_shape_6_2_float
test_func_map["Matmul_shape_2_3_2_shape_2_3_2_float_transpose_b"] = test_Matmul_shape_2_3_2_shape_2_3_2_float_transpose_b
test_func_map["Matmul_shape_2_3_2_shape_2_3_2_uint8_transpose_a"] = test_Matmul_shape_2_3_2_shape_2_3_2_uint8_transpose_a

def test_matmul_op():
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
    test_matmul_op()