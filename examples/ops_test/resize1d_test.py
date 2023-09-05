# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Resize1d_shape_4_2_1_float_nearest_whcn():
    # create graph
    timvx_engine = Engine("test_Resize1d_shape_4_2_1_float_nearest_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [2, 2, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "resize1d"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructResize1dOpConfig(op_name=op_name, type='NEAREST_NEIGHBOR', factor=0.6, 
        align_corners=False, half_pixel_centers=False, target_size=0, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 4]
    input_data_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 2]
    golden_data_list = [1.0, 3.0, 5.0, 7.0,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Resize1d_shape_4_2_1_uint8_nearest_whcn():
    # create graph
    timvx_engine = Engine("test_Resize1d_shape_4_2_1_uint8_nearest_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 2, 1]
    input_scale = 1
    input_zp = 0
    input_quant_info = {}
    input_quant_info["scale"] = input_scale
    input_quant_info["zero_point"] = input_zp
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [2, 2, 1]
    output_scale = 1
    output_zp = 0
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "resize1d"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructResize1dOpConfig(op_name=op_name, type='NEAREST_NEIGHBOR', factor=0.6, 
        align_corners=False, half_pixel_centers=False, target_size=0, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 2, 4]
    input_data_list = [1, 2, 3, 4, 5, 6, 7, 8,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 2]
    golden_data_list = [1, 3, 5, 7,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_Resize1d_shape_5_1_1_float_bilinear_align_corners_whcn():
    # create graph
    timvx_engine = Engine("test_Resize1d_shape_5_1_1_float_bilinear_align_corners_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [5, 1, 1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    output_tensor_shape = [7, 1, 1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "resize1d"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_info = ConstructResize1dOpConfig(op_name=op_name, type='BILINEAR', factor=0.0, 
        align_corners=True, half_pixel_centers=False, target_size=7, op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 1, 5]
    input_data_list = [1.0, 2.0, 3.0, 4.0, 5.0,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 7]
    golden_data_list = [1.0, 1.66666, 2.33333, 3.0, 3.66666, 4.33333, 5.0,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["Resize1d_shape_4_2_1_float_nearest_whcn"] = test_Resize1d_shape_4_2_1_float_nearest_whcn
test_func_map["Resize1d_shape_4_2_1_uint8_nearest_whcn"] = test_Resize1d_shape_4_2_1_uint8_nearest_whcn
test_func_map["Resize1d_shape_5_1_1_float_bilinear_align_corners_whcn"] = test_Resize1d_shape_5_1_1_float_bilinear_align_corners_whcn

def test_resize1d_op():
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
    test_resize1d_op()