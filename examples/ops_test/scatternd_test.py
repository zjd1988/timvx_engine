# -*- coding: utf-8 -*-
import os
import sys
import traceback
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

# setLogLevel("DEBUG")

def test_ScatterND_shape_4_4_4():
    # create graph
    timvx_engine = Engine("test_ScatterND_shape_4_4_4")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    indices_name = "indices"
    indices_tensor_shape = [1, 2]
    assert timvx_engine.create_tensor(indices_name, "INT32", "INPUT", indices_tensor_shape), \
        "construct tensor {} fail!".format(indices_name)

    updates_name = "updates"
    updates_tensor_shape = [4, 4, 2]
    assert timvx_engine.create_tensor(updates_name, "FLOAT32", "INPUT", updates_tensor_shape), \
        "construct tensor {} fail!".format(updates_name)

    output_name = "output"
    output_tensor_shape = [4, 4, 4]
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", output_tensor_shape), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "scatternd"
    op_inputs = ["indices", "updates", ]
    op_outputs = ["output", ]
    op_info = ConstructScatterNDOpConfig(op_name=op_name, shape=[4, 4, 4], op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    indices_np_shape = [2, 1]
    indices_data_list = [0, 2,]
    indices_data = np.array(indices_data_list).reshape(indices_np_shape).astype(np.int32)
    input_dict["indices"] = indices_data

    updates_np_shape = [2, 4, 4]
    updates_data_list = [5, 5, 5, 5, 6, 6, 6, 6,
                         7, 7, 7, 7, 8, 8, 8, 8,
                         1, 1, 1, 1, 2, 2, 2, 2,
                         3, 3, 3, 3, 4, 4, 4, 4,]
    updates_data = np.array(updates_data_list).reshape(updates_np_shape).astype(np.float32)
    input_dict["updates"] = updates_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [4, 4, 4]
    golden_data_list = [5, 5, 5, 5, 6, 6, 6, 6,
                        7, 7, 7, 7, 8, 8, 8, 8,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 2, 2, 2, 2,
                        3, 3, 3, 3, 4, 4, 4, 4,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.float32)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

def test_ScatterND_shape_9():
    # create graph
    timvx_engine = Engine("test_ScatterND_shape_9")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    indices_name = "indices"
    indices_tensor_shape = [4,]
    assert timvx_engine.create_tensor(indices_name, "INT32", "INPUT", indices_tensor_shape), \
        "construct tensor {} fail!".format(indices_name)

    updates_name = "updates"
    updates_tensor_shape = [4,]
    updates_scale = 0.5
    updates_zp = 0
    updates_quant_info = {}
    updates_quant_info["scale"] = updates_scale
    updates_quant_info["zero_point"] = updates_zp
    updates_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(updates_name, "UINT8", "INPUT", updates_tensor_shape, quant_info=updates_quant_info), \
        "construct tensor {} fail!".format(updates_name)

    output_name = "output"
    output_tensor_shape = [9,]
    output_scale = 0.5
    output_zp = 0
    output_quant_info = {}
    output_quant_info["scale"] = output_scale
    output_quant_info["zero_point"] = output_zp
    output_quant_info["quant_type"] = "ASYMMETRIC"
    assert timvx_engine.create_tensor(output_name, "UINT8", "OUTPUT", output_tensor_shape, quant_info=output_quant_info), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "scatternd"
    op_inputs = ["indices", "updates", ]
    op_outputs = ["output", ]
    op_info = ConstructScatterNDOpConfig(op_name=op_name, shape=[9, ], op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    ###################################################################
    ######note: timvx tensor dims is reverse order with np dims########
    ###################################################################
    input_dict = {}
    indices_np_shape = [4,]
    indices_data_list = [4, 3, 1, 7]
    indices_data = np.array(indices_data_list).reshape(indices_np_shape).astype(np.int32)
    input_dict["indices"] = indices_data

    updates_np_shape = [4,]
    updates_data_list = [18, 20, 22, 24]
    updates_data = np.array(updates_data_list).reshape(updates_np_shape).astype(np.uint8)
    input_dict["updates"] = updates_data

    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    output_np_shape = [9, ]
    golden_data_list = [0, 22, 0, 20, 18, 0, 0, 24, 0]
    golden_data = np.array(golden_data_list).reshape(output_np_shape).astype(np.uint8)
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), \
        "check gloden data with output data not equal!\n gloden:{}\n output:{}".format(golden_data, output_data[0])

test_func_map = {}
test_func_map["ScatterND_shape_4_4_4"] = test_ScatterND_shape_4_4_4
test_func_map["ScatterND_shape_9"] = test_ScatterND_shape_9

def test_scatternd_op():
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
    test_scatternd_op()