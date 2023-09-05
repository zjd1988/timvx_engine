# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_Equal_shape_1_uint8():
    # create graph
    timvx_engine = Engine("test_Equal_shape_1_uint8")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    tensor_shape = [1, ]
    input_name1 = "input1"
    input_scale1 = 1
    input_zp1 = 0
    input_quant_info1 = {}
    input_quant_info1["scale"] = input_scale1
    input_quant_info1["zero_point"] = input_zp1
    input_quant_info1["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name1, "UINT8", "INPUT", tensor_shape, quant_info=input_quant_info1), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    input_scale2 = 1
    input_zp2 = 0
    input_quant_info2 = {}
    input_quant_info2["scale"] = input_scale2
    input_quant_info2["zero_point"] = input_zp2
    input_quant_info2["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(input_name2, "UINT8", "INPUT", tensor_shape, quant_info=input_quant_info2), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "BOOL8", "OUTPUT", tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "relational_operations"
    op_inputs = [input_name1, input_name2]
    op_outputs = [output_name, ]
    op_info = ConstructRelationalOperationsOpConfig(op_name=op_name, relational_type='Equal', 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, ]
    input_data_list1 = [255, ]
    input_data1 = np.array(input_data_list1).reshape(input_np_shape).astype(np.uint8)
    input_dict[input_name1] = input_data1

    input_data_list2 = [0, ]
    input_data2 = np.array(input_data_list2).reshape(input_np_shape).astype(np.uint8)
    input_dict[input_name2] = input_data2

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, ]
    golden_output_list = [0, ]
    golden_output_data = np.array(golden_output_list).reshape(output_np_shape).astype(bool)
    assert np.allclose(golden_output_data, output_data[0], atol=1.e-6), \
        "check gloden output data with output data not equal!\n gloden:{}\n output:{}".format(golden_output_data, output_data[0])

def test_NotEqual_shape_5_fp32():
    # create graph
    timvx_engine = Engine("test_NotEqual_shape_5_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    tensor_shape = [5, ]
    input_name1 = "input1"
    assert timvx_engine.create_tensor(input_name1, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert timvx_engine.create_tensor(input_name2, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "BOOL8", "OUTPUT", tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "relational_operations"
    op_inputs = [input_name1, input_name2]
    op_outputs = [output_name, ]
    op_info = ConstructRelationalOperationsOpConfig(op_name=op_name, relational_type='NotEqual', 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [5, ]
    input_data_list1 = [-2.5, -0.1, 0, 0.55, float('inf')]
    input_data1 = np.array(input_data_list1).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name1] = input_data1

    input_data_list2 = [-2, -1, 0.2, 0.55, float('inf')]
    input_data2 = np.array(input_data_list2).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name2] = input_data2

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [5, ]
    golden_output_list = [1, 1, 1, 0, 0]
    golden_output_data = np.array(golden_output_list).reshape(output_np_shape).astype(bool)
    assert np.allclose(golden_output_data, output_data[0], atol=1.e-6), \
        "check gloden output data with output data not equal!\n gloden:{}\n output:{}".format(golden_output_data, output_data[0])

def test_Less_shape_5_1_fp32():
    # create graph
    timvx_engine = Engine("test_Less_shape_5_1_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    tensor_shape = [1, 5,]
    input_name1 = "input1"
    assert timvx_engine.create_tensor(input_name1, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert timvx_engine.create_tensor(input_name2, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "BOOL8", "OUTPUT", tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "relational_operations"
    op_inputs = [input_name1, input_name2]
    op_outputs = [output_name, ]
    op_info = ConstructRelationalOperationsOpConfig(op_name=op_name, relational_type='Less', 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [5, 1,]
    input_data_list1 = [0.1, 0.1, 0, 0.55, float('inf'),]
    input_data1 = np.array(input_data_list1).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name1] = input_data1

    input_data_list2 = [-1, -1, 0.2, 0.55, float('inf'),]
    input_data2 = np.array(input_data_list2).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name2] = input_data2

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [5, 1,]
    golden_output_list = [0, 0, 1, 0, 0]
    golden_output_data = np.array(golden_output_list).reshape(output_np_shape).astype(bool)
    assert np.allclose(golden_output_data, output_data[0], atol=1.e-6), \
        "check gloden output data with output data not equal!\n gloden:{}\n output:{}".format(golden_output_data, output_data[0])

def test_GreaterOrEqual_shape_5_2_1_fp32():
    # create graph
    timvx_engine = Engine("test_GreaterOrEqual_shape_5_2_1_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    tensor_shape = [5, 2, 1]
    input_name1 = "input1"
    assert timvx_engine.create_tensor(input_name1, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert timvx_engine.create_tensor(input_name2, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "BOOL8", "OUTPUT", tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "relational_operations"
    op_inputs = [input_name1, input_name2]
    op_outputs = [output_name, ]
    op_info = ConstructRelationalOperationsOpConfig(op_name=op_name, relational_type='GreaterOrEqual', 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, 2, 5,]
    input_data_list1 = [-2.5, -0.1, 0, 0.55, float('inf'), -2.5, -0.1, 0, 0.55, float('inf')]
    input_data1 = np.array(input_data_list1).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name1] = input_data1

    input_data_list2 = [-2, -1, 0.2, 0.55, float('inf'), -2, -1, 0.2, 0.55, float('inf')]
    input_data2 = np.array(input_data_list2).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name2] = input_data2

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 2, 5,]
    golden_output_list = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    golden_output_data = np.array(golden_output_list).reshape(output_np_shape).astype(bool)
    assert np.allclose(golden_output_data, output_data[0], atol=1.e-6), \
        "check gloden output data with output data not equal!\n gloden:{}\n output:{}".format(golden_output_data, output_data[0])

def test_Greater_shape_5_2_1_1_fp32():
    # create graph
    timvx_engine = Engine("test_Greater_shape_5_2_1_1_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    tensor_shape = [5, 2, 1, 1]
    input_name1 = "input1"
    assert timvx_engine.create_tensor(input_name1, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert timvx_engine.create_tensor(input_name2, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "BOOL8", "OUTPUT", tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "relational_operations"
    op_inputs = [input_name1, input_name2]
    op_outputs = [output_name, ]
    op_info = ConstructRelationalOperationsOpConfig(op_name=op_name, relational_type='Greater', 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, 1, 2, 5,]
    input_data_list1 = [-2.5, -0.1, 0, 0.55, float('inf'), -2.5, -0.1, 0, 0.55, float('inf')]
    input_data1 = np.array(input_data_list1).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name1] = input_data1

    input_data_list2 = [-2, -1, 0.2, 0.55, float('inf'), -2, -1, 0.2, 0.55, float('inf')]
    input_data2 = np.array(input_data_list2).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name2] = input_data2

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 2, 5,]
    golden_output_list = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    golden_output_data = np.array(golden_output_list).reshape(output_np_shape).astype(bool)
    assert np.allclose(golden_output_data, output_data[0], atol=1.e-6), \
        "check gloden output data with output data not equal!\n gloden:{}\n output:{}".format(golden_output_data, output_data[0])

def test_LessOrEqual_shape_1_5_2_1_1_fp32():
    # create graph
    timvx_engine = Engine("test_LessOrEqual_shape_1_5_2_1_1_fp32")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    tensor_shape = [1, 5, 2, 1, 1]
    input_name1 = "input1"
    assert timvx_engine.create_tensor(input_name1, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert timvx_engine.create_tensor(input_name2, "FLOAT32", "INPUT", tensor_shape), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "BOOL8", "OUTPUT", tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "relational_operations"
    op_inputs = [input_name1, input_name2]
    op_outputs = [output_name, ]
    op_info = ConstructRelationalOperationsOpConfig(op_name=op_name, relational_type='LessOrEqual', 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    input_np_shape = [1, 1, 2, 5, 1]
    input_data_list1 = [-2.5, -0.1, 0, 0.55, float('inf'), -2.5, -0.1, 0, 0.55, float('inf')]
    input_data1 = np.array(input_data_list1).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name1] = input_data1

    input_data_list2 = [-2, -1, 0.2, 0.55, float('inf'), -2, -1, 0.2, 0.55, float('inf')]
    input_data2 = np.array(input_data_list2).reshape(input_np_shape).astype(np.float32)
    input_dict[input_name2] = input_data2

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [1, 1, 2, 5, 1]
    golden_output_list = [1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    golden_output_data = np.array(golden_output_list).reshape(output_np_shape).astype(bool)
    assert np.allclose(golden_output_data, output_data[0], atol=1.e-6), \
        "check gloden output data with output data not equal!\n gloden:{}\n output:{}".format(golden_output_data, output_data[0])

test_func_map = {}
test_func_map["Equal_shape_1_uint8"] = test_Equal_shape_1_uint8
test_func_map["NotEqual_shape_5_fp32"] = test_NotEqual_shape_5_fp32
test_func_map["Less_shape_5_1_fp32"] = test_Less_shape_5_1_fp32
test_func_map["GreaterOrEqual_shape_5_2_1_fp32"] = test_GreaterOrEqual_shape_5_2_1_fp32
test_func_map["Greater_shape_5_2_1_1_fp32"] = test_Greater_shape_5_2_1_1_fp32
test_func_map["LessOrEqual_shape_1_5_2_1_1_fp32"] = test_LessOrEqual_shape_1_5_2_1_1_fp32


def test_relational_operations_op():
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
    test_relational_operations_op()