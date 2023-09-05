# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

# FIXME (KC) : There seems to be a limitation that Channel needs to be >= 4,
#              also stride other than 2 is not tested
def test_Reorg_shape_4_4_4_1_u8():
    # create graph
    timvx_engine = Engine("test_Reorg_shape_4_4_4_1_u8")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 4, 4, 1]
    input_scale = 1
    input_zp = 0
    input_quant_info = {}
    input_quant_info["scale"] = input_scale
    input_quant_info["zero_point"] = input_zp
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [2, 2, 16, 1]
    output_scale = 1
    output_zp = 0
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "reorg"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructReorgOpConfig(op_name=op_name, stride=2, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 4, 4, 4]
    input_data_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 16, 2, 2]
    golden_data_list = [0, 2, 4, 6, 16, 18, 20, 22, 0, 2, 4, 6, 16, 18, 20, 22,
                        1, 3, 5, 7, 17, 19, 21, 23, 1, 3, 5, 7, 17, 19, 21, 23,
                        8, 10, 12, 14, 24, 26, 28, 30, 8, 10, 12, 14, 24, 26, 28, 30,
                        9, 11, 13, 15, 25, 27, 29, 31, 9, 11, 13, 15, 25, 27, 29, 31]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Reorg_shape_4_4_4_1_fp32():
    # create graph
    timvx_engine = Engine("test_Reorg_shape_4_4_4_1_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 4, 4, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [2, 2, 16, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "reorg"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructReorgOpConfig(op_name=op_name, stride=2, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 4, 4, 4]
    input_data_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 16, 2, 2]
    golden_data_list = [0, 2, 4, 6, 16, 18, 20, 22, 0, 2, 4, 6, 16, 18, 20, 22,
                        1, 3, 5, 7, 17, 19, 21, 23, 1, 3, 5, 7, 17, 19, 21, 23,
                        8, 10, 12, 14, 24, 26, 28, 30, 8, 10, 12, 14, 24, 26, 28, 30,
                        9, 11, 13, 15, 25, 27, 29, 31, 9, 11, 13, 15, 25, 27, 29, 31]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["Reorg_shape_4_4_4_1_u8"] = test_Reorg_shape_4_4_4_1_u8
test_func_map["Reorg_shape_4_4_4_1_fp32"] = test_Reorg_shape_4_4_4_1_fp32

def test_reorg_op():
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
    test_reorg_op()