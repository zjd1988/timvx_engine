# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_MaxpoolWithArgmax_shape_3_3_1_fp32_kernel_2_stride_2():
    # create graph
    timvx_engine = Engine("test_MaxpoolWithArgmax_shape_3_3_1_fp32_kernel_2_stride_2")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3, 3, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_values_name = "values"
    output_values_shape = [2, 2, 1]
    assert timvx_engine.create_tensor(output_values_name, "FLOAT32", "OUTPUT", output_values_shape), \
        "construct tensor {} fail!".format(output_values_name)

    output_indices_name = "indices"
    output_indices_shape = [2, 2, 1]
    assert timvx_engine.create_tensor(output_indices_name, "UINT8", "OUTPUT", output_indices_shape), \
        "construct tensor {} fail!".format(output_indices_name)

    # construct operations
    op_name = "maxpoolwithargmax"
    op_inputs = ["input", ]
    op_outputs = ["values", "indices", ]
    op_info = ConstructMaxpoolWithArgmaxOpConfig(op_name=op_name, padding='VALID', ksize=[2, 2], stride=[2, 2], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, 3, 3]
    input_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict["input"] = input_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 2]
    golden_values_list = [5, 6, 8, 9,]
    golden_values_data = np.array(golden_values_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_values_data, output_data[0], atol=1.e-6), \
        "check gloden values data with output data not equal!\n gloden:{}\n output:{}".format(golden_values_data, output_data[0])

    golden_indices_list = [3, 2, 1, 0,]
    golden_indices_data = np.array(golden_indices_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_indices_data, output_data[1], atol=1.e-6), \
        "check gloden indices data with output data not equal!\n gloden:{}\n output:{}".format(golden_indices_data, output_data[1])

def test_MaxpoolWithArgmax_shape_4_4_1_uint8_kernel_2_stride_2():
    # create graph
    timvx_engine = Engine("test_MaxpoolWithArgmax_shape_4_4_1_uint8_kernel_2_stride_2")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 4, 1]
    input_scale = 1
    input_zp = 0
    input_quant_info = {}
    input_quant_info["scale"] = input_scale
    input_quant_info["zero_point"] = input_zp
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    output_values_name = "values"
    output_values_shape = [2, 2, 1]
    values_scale = 1
    values_zp = 0
    values_quant_info = {}
    values_quant_info["scale"] = values_scale
    values_quant_info["zero_point"] = values_zp
    values_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_values_name, "UINT8", "OUTPUT", output_values_shape, quant_info=values_quant_info), \
        "construct tensor {} fail!".format(output_values_name)

    output_indices_name = "indices"
    output_indices_shape = [2, 2, 1]
    assert timvx_engine.create_tensor(output_indices_name, "UINT8", "OUTPUT", output_indices_shape), \
        "construct tensor {} fail!".format(output_indices_name)

    # construct operations
    op_name = "maxpoolwithargmax"
    op_inputs = ["input", ]
    op_outputs = ["values", "indices", ]
    op_info = ConstructMaxpoolWithArgmaxOpConfig(op_name=op_name, padding='VALID', ksize=[2, 2], stride=[2, 2], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, 4, 4]
    input_data_list = [1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict["input"] = input_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 2]
    golden_values_list = [5, 6, 11, 12,]
    golden_values_data = np.array(golden_values_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_values_data, output_data[0], atol=1.e-6), \
        "check gloden values data with output data not equal!\n gloden:{}\n output:{}".format(golden_values_data, output_data[0])

    golden_indices_list = [3, 2, 3, 2,]
    golden_indices_data = np.array(golden_indices_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_indices_data, output_data[1], atol=1.e-6), \
        "check gloden indices data with output data not equal!\n gloden:{}\n output:{}".format(golden_indices_data, output_data[1])

test_func_map = {}
test_func_map["MaxpoolWithArgmax_shape_3_3_1_fp32_kernel_2_stride_2"] = test_MaxpoolWithArgmax_shape_3_3_1_fp32_kernel_2_stride_2
test_func_map["MaxpoolWithArgmax_shape_4_4_1_uint8_kernel_2_stride_2"] = test_MaxpoolWithArgmax_shape_4_4_1_uint8_kernel_2_stride_2

def test_maxpoolwithargmax_op():
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
    test_maxpoolwithargmax_op()