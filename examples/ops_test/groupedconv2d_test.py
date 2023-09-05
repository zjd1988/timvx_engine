# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_GroupedConv2d_shape_3_3_6_1_float_group_1_no_bias_whcn():
    # create graph
    timvx_engine = Engine("test_GroupedConv2d_shape_3_3_6_1_float_group_1_no_bias_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3,3,6,1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3,3,6,1]
    weight_data = np.array([-0.50,  0.00, -0.50,
                            -0.50,  0.00, -0.50,
                            -0.50,  0.00, -0.50,
                            1.50,  1.00, -1.50,
                            1.50,  1.00, -1.50,
                            1.50,  1.00, -1.50,
                            -2.50, -2.00, -2.50,
                            -2.50, -2.00, -2.50,
                            -2.50, -2.00, -2.50,
                            3.50,  3.00,  3.50,
                            3.50,  3.00,  3.50,
                            3.50,  3.00,  3.50,
                            -4.50, -4.00, -4.50,
                            -4.50, -4.00, -4.50,
                            -4.50, -4.00, -4.50,
                            -5.50, -5.00,  5.50,
                            -5.50, -5.00,  5.50,
                            -5.50, -5.00,  5.50,]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    output_name = "output"
    output_tensor_shape = [1,1,1,1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "groupedconv2d"
    op_inputs = ["input", "weight"]
    op_outputs = ["output", ]
    op_info = ConstructGroupedConv2dOpConfig(op_name=op_name, stride=[1, 1], dilation=[1, 1],
        grouped_number=1, padding="VALID", op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6, 3, 3]
    input_data_list = [-0.50, -0.50, -0.50,
                        0.00,  1.00,  0.00,
                        0.50,  0.50,  0.50,
                        -1.50, -1.00, -1.00,
                        -0.50,  1.00,  0.50,
                        1.00,  1.00,  1.50,
                        -2.50, -2.00, -2.00,
                        -1.50,  1.50,  1.50,
                        2.00,  2.00,  2.50,
                        -3.50, -3.00, -3.00,
                        -2.50,  2.50,  2.50,
                        3.00,  3.00,  3.50,
                        -4.50, -4.00, -4.00,
                        -3.50,  3.50,  3.50,
                        4.00,  4.00,  4.50,
                        -5.50, -5.00, -5.00,
                        -4.50,  4.50,  4.50,
                        5.00,  5.00,  5.50,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 1, 1]
    golden_data_list = [21.0,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_GroupedConv2d_shape_3_3_6_1_float_group_2_whcn():
    # create graph
    timvx_engine = Engine("test_GroupedConv2d_shape_3_3_6_1_float_group_2_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3,3,6,1]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [3,3,3,2]
    weight_data = np.array([-0.50,  0.00, -0.50,
                            -0.50,  0.00, -0.50,
                            -0.50,  0.00, -0.50,
                            1.50,  1.00, -1.50,
                            1.50,  1.00, -1.50,
                            1.50,  1.00, -1.50,
                            -2.50, -2.00, -2.50,
                            -2.50, -2.00, -2.50,
                            -2.50, -2.00, -2.50,
                            3.50,  3.00,  3.50,
                            3.50,  3.00,  3.50,
                            3.50,  3.00,  3.50,
                            -4.50, -4.00, -4.50,
                            -4.50, -4.00, -4.50,
                            -4.50, -4.00, -4.50,
                            -5.50, -5.00,  5.50,
                            -5.50, -5.00,  5.50,
                            -5.50, -5.00,  5.50,]).reshape(weight_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(weight_name, "FLOAT32", "CONSTANT", weight_tensor_shape, np_data=weight_data), \
        "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [2,]
    bias_data = np.array([-1.25, 1.25,]).reshape(bias_tensor_shape).astype(np.float32)
    assert timvx_engine.create_tensor(bias_name, "FLOAT32", "CONSTANT", bias_tensor_shape, np_data=bias_data), \
        "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [1,1,2,1]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "groupedconv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructGroupedConv2dOpConfig(op_name=op_name, stride=[1, 1], dilation=[1, 1],
        grouped_number=2, padding="VALID", op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6, 3, 3]
    input_data_list = [-0.50, -0.50, -0.50,
                        0.00,  1.00,  0.00,
                        0.50,  0.50,  0.50,
                        -1.50, -1.00, -1.00,
                        -0.50,  1.00,  0.50,
                        1.00,  1.00,  1.50,
                        -2.50, -2.00, -2.00,
                        -1.50,  1.50,  1.50,
                        2.00,  2.00,  2.50,
                        -3.50, -3.00, -3.00,
                        -2.50,  2.50,  2.50,
                        3.00,  3.00,  3.50,
                        -4.50, -4.00, -4.00,
                        -3.50,  3.50,  3.50,
                        4.00,  4.00,  4.50,
                        -5.50, -5.00, -5.00,
                        -4.50,  4.50,  4.50,
                        5.00,  5.00,  5.50,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 1, 1]
    golden_data_list = [-6.25, 27.25,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_GroupedConv2d_shape_3_3_6_1_uint8_group_6_whcn():
    # create graph
    timvx_engine = Engine("test_GroupedConv2d_shape_3_3_6_1_uint8_group_6_whcn")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [3,3,6,1]
    input_scale = 0.5
    input_zp = 10
    input_quant_info = {}
    input_quant_info["scale"] = input_scale
    input_quant_info["zero_point"] = input_zp
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "UINT8", "INPUT", input_tensor_shape, \
        quant_info=input_quant_info), "construct tensor {} fail!".format(input_name)

    weight_name = "weight"
    weight_tensor_shape = [2,2,1,6]
    weight_scale = 0.5
    weight_zp = 9
    weight_quant_info = {}
    weight_quant_info["scale"] = weight_scale
    weight_quant_info["zero_point"] = weight_zp
    weight_quant_info["quant_type"] = "ASYMMETRIC"
    weight_data = np.array([8,  9,
                            8,  9,
                            12, 11,
                            12, 11,
                            4,  5,
                            4,  5,
                            16, 15,
                            16, 15,
                            0, 17,
                            0, 17,
                            6,  5,
                            6, 13,]).reshape(weight_tensor_shape).astype(np.uint8)
    assert timvx_engine.create_tensor(weight_name, "UINT8", "CONSTANT", weight_tensor_shape, \
        quant_info=weight_quant_info, np_data=weight_data), "construct tensor {} fail!".format(weight_name)

    bias_name = "bias"
    bias_tensor_shape = [6,]
    bias_scale = 0.25
    bias_zp = 0
    bias_quant_info = {}
    bias_quant_info["scale"] = bias_scale
    bias_quant_info["zero_point"] = bias_zp
    bias_quant_info["quant_type"] = "ASYMMETRIC"
    bias_data = np.array([-24,-20,-16, 16, -4, 20,]).reshape(bias_tensor_shape).astype(np.int32)
    assert timvx_engine.create_tensor(bias_name, "INT32", "CONSTANT", bias_tensor_shape, \
        quant_info=bias_quant_info, np_data=bias_data), "construct tensor {} fail!".format(bias_name)

    output_name = "output"
    output_tensor_shape = [2,2,6,1]
    output_scale = 0.25
    output_zp = 85
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, \
        quant_info=output_quant_info), "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "groupedconv2d"
    op_inputs = ["input", "weight", "bias"]
    op_outputs = ["output", ]
    op_info = ConstructGroupedConv2dOpConfig(op_name=op_name, stride=[2, 2], dilation=[1, 1],
        grouped_number=6, pad=[0, 1, 0, 1], op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [1, 6, 3, 3]
    input_data_list = [ 9,  9,  9,
                        10, 12, 10,
                        11, 11, 11,
                        7,  8,  8,
                        9, 12, 11,
                        12, 12, 13,
                        5,  6,  6,
                        7, 13, 13,
                        14, 14, 15,
                        3,  4,  4,
                        5, 15, 15,
                        16, 16, 17,
                        1,  2,  2,
                        3, 17, 17,
                        18, 18, 19,
                        3,  0,  0,
                        1, 19, 19,
                        16,  4,  3,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.uint8)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 6, 2, 2]
    golden_data_list = [62, 62,
                        60, 60,
                        53, 62,
                        75, 74,
                        113, 74,
                        33, 44,
                        11, 94,
                        179,150,
                        217, 90,
                        73,  0,
                        229,108,
                        111,126,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["GroupedConv2d_shape_3_3_6_1_float_group_1_no_bias_whcn"] = test_GroupedConv2d_shape_3_3_6_1_float_group_1_no_bias_whcn
test_func_map["GroupedConv2d_shape_3_3_6_1_float_group_2_whcn"] = test_GroupedConv2d_shape_3_3_6_1_float_group_2_whcn
test_func_map["GroupedConv2d_shape_3_3_6_1_uint8_group_6_whcn"] = test_GroupedConv2d_shape_3_3_6_1_uint8_group_6_whcn

def test_groupedconv2d_op():
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
    test_groupedconv2d_op()