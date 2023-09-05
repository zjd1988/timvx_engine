# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_SpatialTransformer_shape_1_3_3_1_u8():
    # create graph
    timvx_engine = Engine("test_SpatialTransformer_shape_1_3_3_1_u8")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [1, 3, 3, 1]
    input_scale = 0.5
    input_zp = 0
    input_quant_info = {}
    input_quant_info["scale"] = input_scale
    input_quant_info["zero_point"] = input_zp
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    theta_name = "theta"
    theta_tensor_shape = [6]
    theta_scale = 0.5
    theta_zp = 0
    theta_quant_info = {}
    theta_quant_info["scale"] = theta_scale
    theta_quant_info["zero_point"] = theta_zp
    theta_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(theta_name, "UINT8", "INPUT", theta_tensor_shape, quant_info=theta_quant_info), \
        "construct tensor {} fail!".format(theta_name)

    output_name = "output"
    output_tensor_shape = [1, 3, 3, 1]
    output_scale = 0.5
    output_zp = 0
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "spatial_transformer"
    op_inputs = ["input", "theta", ]
    op_outputs = ["output", ]
    op_info = ConstructSpatialTransformerOpConfig(op_name=op_name, output_h=3, output_w=3, 
        has_theta_1_1=True, has_theta_1_2=True, has_theta_1_3=True, 
        has_theta_2_1=True, has_theta_2_2=True, has_theta_2_3=True, 
        theta_1_1=1.0, theta_1_2=1.0, theta_1_3=1.0, 
        theta_2_1=1.0, theta_2_2=1.0, theta_2_3=1.0, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, 3, 3, 1]
    input_data_list = [2, 4, 6, 2, 4, 6, 2, 4, 6,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict["input"] = input_data

    theta_np_shape = [6]
    theta_data_list = [2, 2, 2, 2, 2, 2,]
    theta_data = np.array(theta_data_list).reshape(theta_np_shape).astype(np.uint8)
    input_dict["theta"] = theta_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 3, 3, 1]
    golden_data_list = [2, 3, 2, 2, 3, 2, 2, 3, 2,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["SpatialTransformer_shape_1_3_3_1_u8"] = test_SpatialTransformer_shape_1_3_3_1_u8

def test_spatial_transformer_op():
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
    test_spatial_transformer_op()