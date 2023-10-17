# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *
setLogLevel("DEBUG")

def export_simple_add_nbg_graph():
    # test normal graph
    normal_engine = Engine("normal_graph")
    assert normal_engine.create_graph(), "engine create grah fail!"

    input_name1 = "input1"
    assert normal_engine.create_tensor(input_name1, "FLOAT32", "INPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert normal_engine.create_tensor(input_name2, "FLOAT32", "INPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert normal_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(output_name)

    op_name = "simple_add"
    op_inputs = [input_name1, input_name2, ]
    op_outputs = [output_name, ]
    op_info = ConstructEltwiseOpConfig(op_name=op_name, eltwise_type="Add", 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert normal_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # binary_data = normal_engine.compile_to_binary()
    # print("compile binary size is ", len(binary_data))
    assert normal_engine.export_nbg_graph("./simple_add.bin", "./simple_add.json"), "export nbg graph fail"

def test_simple_add_nbg_graph(nbg_graph):
    if type(nbg_graph) == type(""):
        with open(nbg_graph, "rb") as f:
            nbg_data = f.read()
    elif type(nbg_graph) == type(bytearray(b"")):
        nbg_data = nbg_graph
    else:
        assert False, "nbg_graph should be path or binary data, while input type is {}".format(type(nbg_graph))

    binary_engine = Engine("binary_graph")
    assert binary_engine.create_graph(), "engine create grah fail!"

    input_name1 = "input1"
    assert binary_engine.create_tensor(input_name1, "FLOAT32", "INPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert binary_engine.create_tensor(input_name2, "FLOAT32", "INPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert binary_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(output_name)

    op_name = "simple_add"
    op_inputs = [input_name1, input_name2, ]
    op_outputs = [output_name, ]
    op_info = ConstructNBGOpConfig(op_name=op_name, offset=0, length=len(nbg_data), input_count=len(op_inputs),
        output_count=len(op_outputs), op_inputs=op_inputs, op_outputs=op_outputs)
    np_data = np.frombuffer(nbg_data, dtype=np.int8)
    assert binary_engine.create_operation(op_info, np_data), "construct operation {} fail!".format(op_name)

    input_dict = {}
    input_data1_shape = [1, 1, 1, 1]
    input_data1 = np.array([1,]).reshape(input_data1_shape).astype(np.float32)
    input_data2_shape = [1, 1, 1, 1]
    input_data2 = np.array([1,]).reshape(input_data2_shape).astype(np.float32)
    input_dict["input1"] = input_data1
    input_dict["input2"] = input_data2

    output_data = binary_engine.run_graph(input_dict)

    output_shape = [1, 1, 1, 1]
    output_data_list = [2,]
    golden_data = np.array(output_data_list).reshape(output_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with binary_engine output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])
    print("binary_engine output:\n", output_data[0])

def test_gen_binary_graph_with_simple_add():
    # test normal graph
    normal_engine = Engine("normal_graph")
    assert normal_engine.create_graph(), "engine create grah fail!"

    input_name1 = "input1"
    assert normal_engine.create_tensor(input_name1, "FLOAT32", "INPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(input_name1)

    input_name2 = "input2"
    assert normal_engine.create_tensor(input_name2, "FLOAT32", "INPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(input_name2)

    output_name = "output"
    assert normal_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", [1, 1, 1, 1]), \
        "construct tensor {} fail!".format(output_name)

    op_name = "simple_add"
    op_inputs = [input_name1, input_name2, ]
    op_outputs = [output_name, ]
    op_info = ConstructEltwiseOpConfig(op_name=op_name, eltwise_type="Add", 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert normal_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    binary_data = normal_engine.compile_to_binary()
    assert 0 != len(binary_data), "compile to binary fail"

    input_data1 = np.array
    input_dict = {}
    input_data1_shape = [1, 1, 1, 1]
    input_data1 = np.array([1,]).reshape(input_data1_shape).astype(np.float32)
    input_data2_shape = [1, 1, 1, 1]
    input_data2 = np.array([1,]).reshape(input_data2_shape).astype(np.float32)
    input_dict["input1"] = input_data1
    input_dict["input2"] = input_data2

    output_data = normal_engine.run_graph(input_dict)

    output_shape = [1, 1, 1, 1]
    output_data_list = [2,]
    golden_data = np.array(output_data_list).reshape(output_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with normal_engine output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])
    print("normal_engine output:\n", output_data[0])

    # test binary graph
    test_simple_add_nbg_graph(binary_data)

if __name__ == "__main__":
    # 1 export simple add NBG graph
    # export_simple_add_nbg_graph()
    # 2 test simple add NBG graph binary file
    # test_simple_add_nbg_graph("./simple_add.bin")
    # 3 test NBG graph from binary data
    test_gen_binary_graph_with_simple_add()