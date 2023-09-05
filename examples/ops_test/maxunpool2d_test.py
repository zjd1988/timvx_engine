# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_MaxUnpool2d_shape_2_2_1_fp32_kernel_2_stride_2():
    # create graph
    timvx_engine = Engine("test_MaxUnpool2d_shape_2_2_1_fp32_kernel_2_stride_2")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_values_name = "values"
    input_values_shape = [2, 2, 1]
    assert timvx_engine.create_tensor(input_values_name, "FLOAT32", "INPUT", input_values_shape), \
        "construct tensor {} fail!".format(input_values_name)

    input_indices_name = "indices"
    input_indices_shape = [2, 2, 1]
    assert timvx_engine.create_tensor(input_indices_name, "UINT8", "INPUT", input_indices_shape), \
        "construct tensor {} fail!".format(input_indices_name)

    output_tensor_name = "output"
    output_tensor_shape = [3, 3, 1]
    assert timvx_engine.create_tensor(output_tensor_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_tensor_name)

    # construct operations
    op_name = "maxunpool2d"
    op_inputs = ["values", "indices", ]
    op_outputs = ["output", ]
    op_info = ConstructMaxUnpool2dOpConfig(op_name=op_name, ksize=[2, 2], stride=[2, 2], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    values_np_shape = [1, 2, 2]
    values_data_list = [5, 6, 8, 9]
    values_data = np.array(values_data_list).reshape(values_np_shape).astype(np.float32)
    input_dict["values"] = values_data

    indices_np_shape = [1, 2, 2]
    indices_data_list = [3, 2, 1, 0]
    indices_data = np.array(indices_data_list).reshape(indices_np_shape).astype(np.uint8)
    input_dict["indices"] = indices_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 3, 3]
    golden_output_list = [0, 0, 0, 0, 5, 6, 0, 8, 9]
    golden_output_data = np.array(golden_output_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_output_data, output_data[0], atol=1.e-6), \
        "check gloden output data with output data not equal!\n gloden:{}\n output:{}".format(golden_output_data, output_data[0])

def test_MaxUnpool2d_shape_2_2_1_uint8_kernel_2_stride_2():
    # create graph
    timvx_engine = Engine("test_MaxUnpool2d_shape_2_2_1_uint8_kernel_2_stride_2")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_values_name = "values"
    input_values_shape = [2, 2, 1]
    values_scale = 1
    values_zp = 0
    values_quant_info = {}
    values_quant_info["scale"] = values_scale
    values_quant_info["zero_point"] = values_zp
    values_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_values_name, "UINT8", "INPUT", input_values_shape, quant_info=values_quant_info), \
        "construct tensor {} fail!".format(input_values_name)

    input_indices_name = "indices"
    input_indices_shape = [2, 2, 1]
    assert timvx_engine.create_tensor(input_indices_name, "UINT8", "INPUT", input_indices_shape), \
        "construct tensor {} fail!".format(input_indices_name)

    output_tensor_name = "output"
    output_tensor_shape = [4, 4, 1]
    output_scale = 1
    output_zp = 0
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_tensor_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_tensor_name)

    # construct operations
    op_name = "maxunpool2d"
    op_inputs = ["values", "indices", ]
    op_outputs = ["output", ]
    op_info = ConstructMaxUnpool2dOpConfig(op_name=op_name, ksize=[2, 2], stride=[2, 2], 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    values_np_shape = [1, 2, 2]
    values_data_list = [5, 6, 11, 12]
    values_data = np.array(values_data_list).reshape(values_np_shape).astype(np.uint8)
    input_dict["values"] = values_data

    indices_np_shape = [1, 2, 2]
    indices_data_list = [3, 2, 3, 2]
    indices_data = np.array(indices_data_list).reshape(indices_np_shape).astype(np.uint8)
    input_dict["indices"] = indices_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 4, 4]
    golden_output_list = [0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 11, 12, 0]
    golden_output_data = np.array(golden_output_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_output_data, output_data[0], atol=1.e-6), \
        "check gloden output data with output data not equal!\n gloden:{}\n output:{}".format(golden_output_data, output_data[0])

test_func_map = {}
test_func_map["MaxUnpool2d_shape_2_2_1_fp32_kernel_2_stride_2"] = test_MaxUnpool2d_shape_2_2_1_fp32_kernel_2_stride_2
test_func_map["MaxUnpool2d_shape_2_2_1_uint8_kernel_2_stride_2"] = test_MaxUnpool2d_shape_2_2_1_uint8_kernel_2_stride_2

def test_maxunpool2d_op():
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
    test_maxunpool2d_op()