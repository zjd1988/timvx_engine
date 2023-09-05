# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Unstack_shape_4_3_axis_0():
    # create graph
    timvx_engine = Engine("test_Unstack_shape_4_3_axis_0")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 3]
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape), \
        "construct tensor {} fail!".format(input_name)

    output_name1 = "output1"
    output_tensor_shape1 = [3]
    assert timvx_engine.create_tensor(output_name1, "FLOAT32", "OUTPUT", output_tensor_shape1), \
        "construct tensor {} fail!".format(output_name1)

    output_name2 = "output2"
    output_tensor_shape2 = [3]
    assert timvx_engine.create_tensor(output_name2, "FLOAT32", "OUTPUT", output_tensor_shape2), \
        "construct tensor {} fail!".format(output_name2)

    output_name3 = "output3"
    output_tensor_shape3 = [3]
    assert timvx_engine.create_tensor(output_name3, "FLOAT32", "OUTPUT", output_tensor_shape3), \
        "construct tensor {} fail!".format(output_name3)

    output_name4 = "output4"
    output_tensor_shape4 = [3]
    assert timvx_engine.create_tensor(output_name4, "FLOAT32", "OUTPUT", output_tensor_shape4), \
        "construct tensor {} fail!".format(output_name4)

    # construct operations
    op_name = "unstack"
    op_inputs = ["input", ]
    op_outputs = ["output1", "output2", "output3", "output4", ]
    op_info = ConstructUnstackOpConfig(op_name=op_name, axis=0, output_num=4, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [3, 4]
    input_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [3]
    golden_data_list1 = [1, 5, 9]
    golden_data1 = np.array(golden_data_list1).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data1, output_data[0], atol=1.e-6), \
        "check gloden data1 with output data not equal!\n gloden:{}\n output:{}".format(golden_data1, output_data[0])

    golden_data_list2 = [2, 6, 10]
    golden_data2 = np.array(golden_data_list2).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data2, output_data[1], atol=1.e-6), \
        "check gloden data2 with output data not equal!\n gloden:{}\n output:{}".format(golden_data2, output_data[1])

    golden_data_list3 = [3, 7, 11]
    golden_data3 = np.array(golden_data_list3).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data3, output_data[2], atol=1.e-6), \
        "check gloden data3 with output data not equal!\n gloden:{}\n output:{}".format(golden_data3, output_data[2])

    golden_data_list4 = [4, 8, 12]
    golden_data4 = np.array(golden_data_list4).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data4, output_data[3], atol=1.e-6), \
        "check gloden data4 with output data not equal!\n gloden:{}\n output:{}".format(golden_data4, output_data[3])

def test_Unstack_shape_4_3_axis_1():
    # create graph
    timvx_engine = Engine("test_Unstack_shape_4_3_axis_1")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    input_tensor_shape = [4, 3]
    input_scale = 0.5
    input_zp = 0
    input_quant_info = {}
    input_quant_info["scale"] = input_scale
    input_quant_info["zero_point"] = input_zp
    input_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", input_tensor_shape, quant_info=input_quant_info), \
        "construct tensor {} fail!".format(input_name)

    output_name1 = "output1"
    output_tensor_shape1 = [4]
    output_scale1 = 0.5
    output_zp1 = 0
    output_quant_info1 = {}
    output_quant_info1["scale"] = output_scale1
    output_quant_info1["zero_point"] = output_zp1
    output_quant_info1["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name1, "FLOAT32", "OUTPUT", output_tensor_shape1, quant_info=output_quant_info1), \
        "construct tensor {} fail!".format(output_name1)

    output_name2 = "output2"
    output_tensor_shape2 = [4]
    output_scale2 = 0.5
    output_zp2 = 0
    output_quant_info2 = {}
    output_quant_info2["scale"] = output_scale2
    output_quant_info2["zero_point"] = output_zp2
    output_quant_info2["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name2, "FLOAT32", "OUTPUT", output_tensor_shape2, quant_info=output_quant_info2), \
        "construct tensor {} fail!".format(output_name2)

    output_name3 = "output3"
    output_tensor_shape3 = [4]
    output_scale3 = 0.5
    output_zp3 = 0
    output_quant_info3 = {}
    output_quant_info3["scale"] = output_scale3
    output_quant_info3["zero_point"] = output_zp3
    output_quant_info3["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name3, "FLOAT32", "OUTPUT", output_tensor_shape3, quant_info=output_quant_info3), \
        "construct tensor {} fail!".format(output_name3)

    # construct operations
    op_name = "unstack"
    op_inputs = ["input", ]
    op_outputs = ["output1", "output2", "output3", ]
    op_info = ConstructUnstackOpConfig(op_name=op_name, axis=1, output_num=3, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_np_shape = [3, 4]
    input_data_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    input_data = np.array(input_data_list).reshape(input_np_shape).astype(np.float32)
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [4]
    golden_data_list1 = [2, 4, 6, 8]
    golden_data1 = np.array(golden_data_list1).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data1, output_data[0], atol=1.e-6), \
        "check gloden data1 with output data not equal!\n gloden:{}\n output:{}".format(golden_data1, output_data[0])

    golden_data_list2 = [10, 12, 14, 16]
    golden_data2 = np.array(golden_data_list2).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data2, output_data[1], atol=1.e-6), \
        "check gloden data2 with output data not equal!\n gloden:{}\n output:{}".format(golden_data2, output_data[1])

    golden_data_list3 = [18, 20, 22, 24]
    golden_data3 = np.array(golden_data_list3).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data3, output_data[2], atol=1.e-6), \
        "check gloden data3 with output data not equal!\n gloden:{}\n output:{}".format(golden_data3, output_data[2])

test_func_map = {}
test_func_map["Unstack_shape_4_3_axis_0"] = test_Unstack_shape_4_3_axis_0
test_func_map["Unstack_shape_4_3_axis_1"] = test_Unstack_shape_4_3_axis_1

def test_unstack_op():
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
    test_unstack_op()