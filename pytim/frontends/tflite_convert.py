# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
from functools import reduce
from .tflite_parser import parse_tflite_model
from ..timvx import *


class Tflite2TimVxEngine():
    def __init__(self):
        self.op_construct_funcs = {}
        self.tflite_op_timvx_op_map = {}
        self.register('ELU', 'Activation', self.construct_activation_op)
        self.register('RELU', 'Activation', self.construct_activation_op)
        self.register('RELU_N1_TO_1', 'Activation', self.construct_activation_op)
        self.register('RELU6', 'Activation', self.construct_activation_op)
        self.register('TANH', 'Activation', self.construct_activation_op)
        self.register("LOGISTIC", 'Activation', self.construct_activation_op)
        self.register("PRELU", 'Activation', self.construct_activation_op)
        self.register('ADD', 'Eltwise', self.construct_eltwise_op)
        self.register('SUB', 'Eltwise', self.construct_eltwise_op)
        self.register('MUL', 'Eltwise', self.construct_eltwise_op)
        self.register('DIV', 'Eltwise', self.construct_eltwise_op)
        self.register('CONV_2D', 'Conv2d', self.construct_conv2d_op)
        # self.register('VARIABLE', 'Variable', construct_variable_op)
        self.register('RESHAPE', 'Reshape', self.construct_reshape_op)
        self.register('TRANSPOSE', 'Transpose', self.construct_transpose_op)
        self.register('RESIZE_BILINEAR', 'Resize', self.construct_resize_op)
        self.register('RESIZE_NEAREST_NEIGHBOR', 'Resize', self.construct_resize_op)
        self.register("MAX_POOL_2D", "Pool2d", self.construct_pool2d_op)
        self.register("AVERAGE_POOL_2D", "Pool2d", self.construct_pool2d_op)
        self.register("FULLY_CONNECTED", "FullyConnected", self.construct_fullyconnected_op)
        self.register("CONCATENATION", "Concat", self.construct_concat_op)
        self.register("SOFTMAX", "Softmax", self.construct_softmax_op)
        self.register("QUANTIZE", "DataConvert", self.consturct_dataconvert_op)
        self.register("DEQUANTIZE", "DataConvert", self.consturct_dataconvert_op)


    def convert_axis(self, axis_in, dim_num):
        return dim_num - (dim_num + axis_in if axis_in < 0 else axis_in) - 1


    def insert_new_tensor(self, tflite_model_info, tensor_name, tensor_shape, tensor_type, quant_info={}, tensor_buffer=None):
        insert_tensor = {}
        tensors_info = tflite_model_info["tensors"]
        insert_tensor_index = len(tensors_info)
        insert_tensor["name"] = tensor_name + "_" + str(insert_tensor_index)
        insert_tensor["shape"] = tensor_shape
        insert_tensor["type"] = tensor_type
        insert_tensor["quantization"] = quant_info
        insert_tensor["buffer"] = None
        if isinstance(tensor_buffer, np.ndarray):
            insert_tensor["buffer"] = tensor_buffer
        tflite_model_info["tensors"].append(insert_tensor)
        return insert_tensor_index


    def insert_new_node(self, tflite_model_info, op_name, op_type, op_inputs, op_outputs, node_attr={}):
        insert_node = {}
        insert_nodes_info = tflite_model_info["insert_nodes"]
        insert_node_index = len(insert_nodes_info)
        insert_node["name"] = op_name + "_" + str(op_type) + "_" + str(insert_node_index)
        insert_node["type"] = op_type
        insert_node["inputs"] = op_inputs
        insert_node["outputs"] = op_outputs
        insert_node["attr"] = node_attr
        tflite_model_info["insert_nodes"].append(insert_node)
        return insert_node_index


    def get_timvx_transpose_perm(self, perm):
        perm_out = copy.deepcopy(perm)
        perm_out.reverse()
        perm_in = []
        ovx_perm = []
        perm_size = len(perm)
        for i in range(perm_size):
            perm_in.append(perm_size - 1 - i)
        for i in range(len(perm_out)):
            for j in range(len(perm_in)):
                if perm_out[i] == perm_in[j]:
                    ovx_perm.append(j)
                    break
        return ovx_perm


    def get_tensor_actual_shape(self, tensor_info, perm=[]):
        tensor_shape = copy.deepcopy(tensor_info["shape"])
        tensor_shape.reverse()
        return tensor_shape


    def convert_tflite_activation_type_to_timvx_type(self, tflite_activation_type):
        if "RELU" == tflite_activation_type:
            return "Relu"
        elif "RELU_N1_TO_1" == tflite_activation_type:
            return "Relu1"
        elif "RELU6" == tflite_activation_type:
            return "Relu6"
        elif "TANH" == tflite_activation_type:
            return "Tanh"
        else:
            assert False, "not support {} activation type".format(tflite_activation_type)


    def get_node_inputs_outputs_name(self, node_info, tensors_info):
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_inputs_name = []
        op_outputs_name = []
        for i in range(len(inputs)):
            op_inputs_name.append(tensors_info[inputs[i]]["name"])
        for i in range(len(outputs)):
            op_outputs_name.append(tensors_info[outputs[i]]["name"])
        return op_inputs_name, op_outputs_name


    def insert_fused_activation_function(self, tflite_model_info, node_index):
        tensors_info = tflite_model_info["tensors"]
        node_info = tflite_model_info["nodes"][node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        fused_activation_function = node_info["attr"]["fused_activation_function"]
        activation_tensor_name = tensors_info[outputs[0]]["name"]
        activation_tensor_shape = tensors_info[outputs[0]]["shape"]
        activation_tensor_type = tensors_info[outputs[0]]["type"]
        activation_tensor_quant = tensors_info[outputs[0]]["quantization"]
        activation_tensor_index = self.insert_new_tensor(tflite_model_info, activation_tensor_name, 
            activation_tensor_shape, activation_tensor_type, activation_tensor_quant)

        activation_op_name = op_name
        # activation_op_type = self.convert_tflite_activation_type_to_timvx_type(fused_activation_function)
        activation_op_type = fused_activation_function
        activation_op_inputs = [activation_tensor_index, ]
        activation_op_outputs = copy.deepcopy(outputs)
        outputs[0] = activation_tensor_index
        activation_node_index = self.insert_new_node(tflite_model_info, activation_op_name, 
            activation_op_type, activation_op_inputs, activation_op_outputs)
        return activation_node_index


    def construct_activation_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 1, "tflite {} op should have 1 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)
        parameter = {}
        if tflite_op_type == "ELU":
            activation_type = "Elu"
        elif tflite_op_type == "RELU":
            activation_type = "Relu"
        elif tflite_op_type == "RELU_N1_TO_1":
            activation_type = "Relu1"
        elif tflite_op_type == "RELU6":
            activation_type = "Relu6"
        elif tflite_op_type == "LOGISTIC":
            activation_type = "Sigmoid"
        elif tflite_op_type == "PRELU":
            activation_type = "Prelu"
            parameter["axis"] = 0
        else:
            assert False, "unspppoted activation type {}".format(tflite_op_type)

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
        op_info = ConstructActivationOpConfig(op_name=op_name, activation_type=activation_type, op_inputs=op_inputs, 
            op_outputs=op_outputs)

        if log_flag:
            print("construct {} op with info:\n{}".format(op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
        return op_info


    def construct_eltwise_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        pass


    def construct_resize_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 2, "tflite {} op should have 2 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)

        size_tensor = tensors_info[inputs[1]]
        assert "buffer" in size_tensor.keys(), "tflite {} op second input should be const tensor".format(tflite_op_type)
        size = size_tensor["buffer"].tolist()
        op_attr["size"] = size
        node_info["inputs"] = inputs[0:1]

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)


    def construct_concat_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) > 1, "tflite {} op should have more than 1 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)

        tensor_shape = tensors_info[inputs[0]]["shape"]
        axis = op_attr["axis"]
        timvx_axis = self.convert_axis(axis_in, len(tensor_shape))

        none_zero_inputs = []
        for i in range(len(inputs)):
            tensor_shape = tensors_info[inputs[i]]["shape"]
            element_num = reduce(lambda x, y:x*y, tensor_shape)
            if element_num != 0:
                none_zero_inputs.append(inputs[i])
            else:
                print("Remove zero sized tensor {} from concat's input list".format())
        node_info["inputs"] = none_zero_inputs

        activation_node_index = -1
        fused_activation_function = op_attr["fused_activation_function"]
        if fused_activation_function != "NONE":
            activation_node_index = self.insert_fused_activation_function(tflite_model_info, node_index)

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
        op_info = ConstructConcatOpConfig(op_name=op_name, axis=timvx_axis, op_inputs=op_inputs, op_outputs=op_outputs)
        if log_flag:
            print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

        if activation_node_index != -1:
            self.construct_node(tflite_model_info, activation_node_index, engine, log_flag=log_flag, insert_flag=True)
        return op_info


    def construct_conv2d_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 2 or len(inputs) == 3, "tflite {} op should have 2 or 3 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)
        input_tensor = tensors_info[inputs[0]]
        weight_tensor = tensors_info[inputs[1]]
        input_shape = self.get_tensor_actual_shape(input_tensor)
        weight_shape = self.get_tensor_actual_shape(weight_tensor)
        #  input layout CWHN, weight layout IWHO
        groups = input_shape[0] // weight_shape[0]
        weights = weight_shape[3]
        kernel_h = weight_shape[2]
        kernel_w = weight_shape[1]

        activation_node_index = -1
        fused_activation_function = op_attr["fused_activation_function"]
        if fused_activation_function != "NONE":
            activation_node_index = self.insert_fused_activation_function(tflite_model_info, node_index)

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        if input_shape[0] == weight_shape[0]:
            op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
            op_info = ConstructConv2dOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs, 
                weights=weights, padding=op_attr["padding"], ksize=[kernel_w, kernel_h], 
                stride=[op_attr["stride_w"], op_attr["stride_w"]], 
                dilation=[op_attr["dilation_w_factor"], op_attr["dilation_h_factor"]],
                multiplier=0, input_layout="CWHN")
        else:
            op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
            op_info = ConstructGroupedConv2dOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs, 
                padding=op_attr["padding"], stride=[op_attr["stride_w"], op_attr["stride_w"]], 
                dilation=[op_attr["dilation_w_factor"], op_attr["dilation_h_factor"]],
                grouped_number=groups, input_layout="CWHN")
        if log_flag:
            print("construct {} op with info:\n{}".format(op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

        if activation_node_index != -1:
            self.construct_node(tflite_model_info, activation_node_index, engine, log_flag=log_flag, insert_flag=True)
        return op_info


    def construct_pool2d_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 1, "tflite {} op should have 1 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)

        activation_node_index = -1
        fused_activation_function = op_attr["fused_activation_function"]
        if fused_activation_function != "NONE":
            activation_node_info = self.insert_fused_activation_function(tflite_model_info, node_index)

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        if tflite_op_type == "MAX_POOL_2D":
            op_type = "MAX"
        elif tflite_op_type == "AVERAGE_POOL_2D":
            op_type = "AVG"
        else:
            assert False, "current not support {} pool type".format(op_type)
        op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
        op_info = ConstructPool2dOpConfig(op_name, op_type, ksize=[op_attr["filter_width"], op_attr["filter_height"]], 
            stride=[op_attr["stride_w"], op_attr["stride_h"]], padding=op_attr["padding"], round_type="FLOOR", 
            layout="CWHN", op_inputs=op_inputs, op_outputs=op_outputs)

        if log_flag:
            print("construct {} op with info:\n{}".format(op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

        if activation_node_index != -1:
            self.construct_node(tflite_model_info, activation_node_index, engine, log_flag=log_flag, insert_flag=True)
        return op_info


    def construct_fullyconnected_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 2 or len(inputs) == 3, "tflite {} op should have 2 or 3 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)
        input_tensor = tensors_info[inputs[0]]
        weight_tensor = tensors_info[inputs[1]]
        input_shape = self.get_tensor_actual_shape(input_tensor)
        weight_shape = self.get_tensor_actual_shape(weight_tensor)

        if len(input_shape) > 2 or (len(input_shape) == 2 and input_shape[0] != weight_shape[0]):
            input_size = weight_shape[0]
            total_input_size = 1
            for i in range(len(input_shape)):
                total_input_size *= input_shape[i]
            input_batch = total_input_size // input_size
            new_shape = [input_batch, input_size]
            reshape_output_tensor_info = copy.deepcopy(input_tensor)
            reshape_output_tensor_info["shape"] = new_shape
            reshape_output_tensor_index = self.insert_new_tensor(tflite_model_info, reshape_output_tensor_info["name"], 
                reshape_output_tensor_info["shape"], reshape_output_tensor_info["type"], 
                reshape_output_tensor_info["quantization"])

            reshape_op_inputs = [inputs[0], ]
            reshape_op_outputs = [reshape_output_tensor_index, ]
            reshape_op_attr = {"new_shape":new_shape}
            reshape_node_index = self.insert_new_node(tflite_model_info, op_name, "RESHAPE", 
                reshape_op_inputs, reshape_op_outputs, reshape_op_attr)
            self.construct_node(tflite_model_info, reshape_node_index, engine, log_flag=log_flag, insert_flag=True)
            inputs[0] = reshape_output_tensor_index

        activation_node_index = -1
        fused_activation_function = op_attr["fused_activation_function"]
        if fused_activation_function != "NONE":
            activation_node_index = self.insert_fused_activation_function(tflite_model_info, node_index)

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        axis = 0
        weights = weight_shape[1]
        op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
        op_info = ConstructFullyConnectedOpConfig(op_name=op_name, axis=axis, weights=weights, 
            op_inputs=op_inputs, op_outputs=op_outputs)
        if log_flag:
            print("construct {} op with info:\n{}".format(op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

        if activation_node_index != -1:
            self.construct_node(tflite_model_info, activation_node_index, engine, log_flag=log_flag, insert_flag=True)
        return op_info


    def construct_transpose_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 2, "tflite {} op should have 2 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)

        perm_tensor = tensors_info[inputs[1]]
        assert "buffer" in perm_tensor.keys(), "tflite {} op second input should be const tensor".format(tflite_op_type)
        perm = perm_tensor["buffer"].tolist()
        ovx_perm = self.get_timvx_transpose_perm(perm)
        node_info["inputs"] = inputs[0:1]

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
        op_info = ConstructTransposeOpConfig(op_name=op_name, perm=ovx_perm, op_inputs=op_inputs, op_outputs=op_outputs)
        if log_flag:
            print("construct {} op with info:\n{}".format(op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
        return op_info


    def construct_reshape_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 1 or len(inputs) == 2, "tflite {} op should have 1 or 2 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)
        new_shape = []
        total_size = 1
        negative_index = 0
        no_nagetive_shape = []
        do_shape_inference = False
        input_tensor = tensors_info[inputs[0]]
        if len(inputs) == 2:
            shape_tensor = tensors_info[inputs[1]]
            if shape_tensor["type"] == "INT32" and shape_tensor["shape"] == 1 and "buffer" in shape_tensor.keys():
                new_shape = shape_tensor["buffer"].tolist()
                op_attr["new_shape"] = new_shape
        else:
            assert "new_shape" in node_info["attr"].keys(), "tflite {} op'attr should have new_shape item".format(tflite_op_type)
            new_shape = copy.deepcopy(node_info["attr"]["new_shape"])
            new_shape.reverse()

        for i in range(len(input_tensor["shape"])):
            total_size *= input_tensor["shape"][i]

        for i in range(len(new_shape)):
            if new_shape[i] != -1:
                total_size /= new_shape[i]
                no_nagetive_shape.append(new_shape[i])
            else:
                do_shape_inference = True
                negative_index = i
                no_nagetive_shape.append(0)  # hold a place for changes to the value
        if do_shape_inference:
            no_nagetive_shape[negative_index] = total_size

        node_info["inputs"] = [inputs[0], ]
        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
        op_info = ConstructReshapeOpConfig(op_name=op_name, size=no_nagetive_shape, op_inputs=op_inputs, op_outputs=op_outputs)
        if log_flag:
            print("construct {} op with info:\n{}".format(op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
        return op_info


    def construct_softmax_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 1, "tflite {} op should have 1 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)
        assert "beta" in node_info["attr"].keys(), "tflite {} op'attr should have beta item"

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        beta = node_info["attr"]["beta"]
        axis = 0
        op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
        op_info = ConstructSoftmaxOpConfig(op_name=op_name, beta=beta, axis=axis, op_inputs=op_inputs, op_outputs=op_outputs)
        if log_flag:
            print("construct {} op with info:\n{}".format(op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
        return op_info


    def consturct_dataconvert_op(self, tflite_op_type, tflite_model_info, node_index, engine, log_flag, insert_flag):
        tensors_info = tflite_model_info["tensors"]
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        op_name = node_info["name"]
        op_attr = node_info["attr"]
        assert len(inputs) == 1, "tflite {} op should have 1 input".format(tflite_op_type)
        assert len(outputs) == 1, "tflite {} op should have 1 output".format(tflite_op_type)

        self.construct_node_tensors(tflite_model_info, node_index, engine, log_flag=log_flag, insert_flag=insert_flag)

        op_inputs, op_outputs = self.get_node_inputs_outputs_name(node_info, tensors_info)
        op_info = ConstructSimpleOperationsOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs)
        if log_flag:
            print("construct {} op with info:\n{}".format(op_name, op_info))
        assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
        return op_info


    def register(self, tflite_op_type, timvx_op_type, op_func):
        if tflite_op_type not in self.tflite_op_timvx_op_map.keys():
            self.tflite_op_timvx_op_map[tflite_op_type] = timvx_op_type
            self.op_construct_funcs[timvx_op_type] = op_func
        else:
            print("already register {}".format(tflite_op_type))


    def construct_node_tensors(self, tflite_model_info, node_index, engine, tensor_index_list=[], log_flag=False, insert_flag=False):
        nodes_info = tflite_model_info["nodes"] if insert_flag == False else tflite_model_info["insert_nodes"]
        node_info = nodes_info[node_index]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        index_list = []
        if len(tensor_index_list) != 0:
            index_list = tensor_index_list
        else:
            index_list = inputs + outputs
        for i in range(len(index_list)):
            tensor_index = index_list[i]
            self.construct_engine_tensor(engine, tflite_model_info, tensor_index, log_flag)


    def construct_node(self, tflite_model_info, node_index, engine, log_flag = False, insert_flag=False):
        if insert_flag == True:
            node_info = tflite_model_info["insert_nodes"][node_index]
        else:
            node_info = tflite_model_info["nodes"][node_index]
        tflite_op_type = node_info['type']
        if tflite_op_type not in self.tflite_op_timvx_op_map.keys():
            assert False, "have not register {}".format(tflite_op_type)
        else:
            timvx_op_type = self.tflite_op_timvx_op_map[tflite_op_type]
            node_info = self.op_construct_funcs[timvx_op_type](tflite_op_type, tflite_model_info, 
                node_index, engine, log_flag, insert_flag)
            return node_info


    def create_tensor(self, engine:Engine, tensor_name:str, tensor_attr:str, tflite_tensor:dict, log_flag:bool=False):
        quant_info = {}
        timvx_tensor_shape = []
        np_data = np.array([])
        if "shape" in tflite_tensor.keys():
            tflite_tensor_shape = tflite_tensor["shape"]
            timvx_tensor_shape = self.get_tensor_actual_shape(tflite_tensor)
        tensor_dtype = tflite_tensor["type"]
        if "quantization" in tflite_tensor.keys() and None != tflite_tensor["quantization"]:
            qnt_type = "ASYMMETRIC"
            tensor_quant = tflite_tensor["quantization"]
            assert len(tensor_quant["scale"]) == len(tensor_quant["zero_point"]), \
                "tensor {} have unequal scale/zero_point size"
            scale = tensor_quant["scale"]
            zero_point = tensor_quant["zero_point"]
            quantized_dimension = tensor_quant["quantized_dimension"]
            if len(scale) == 1:
                quant_info["scale"] = scale[0]
                quant_info["zero_point"] = zero_point[0]
            else:
                qnt_type = "SYMMETRIC_PER_CHANNEL"
                quant_info["scale"] = copy.deepcopy(scale)
                quant_info["zero_point"] = copy.deepcopy(zero_point)
                if quantized_dimension != -1:
                    channel_dim = quantized_dimension
                    vx_channel_dim = self.convert_axis(channel_dim, len(tflite_tensor_shape))
                    quant_info["channel_dim"] = vx_channel_dim
        else:
            qnt_type = "NONE"
        quant_info["quant_type"] = qnt_type
        if "buffer" in tflite_tensor.keys() and isinstance(tflite_tensor["buffer"], np.ndarray):
            np_data = tflite_tensor["buffer"]
        # if log_flag:
        #     print("********************************")
        #     print("construct tensor {} with:".format(tensor_name))
        #     print("tensor type    : {}".format(tensor_dtype))
        #     print("tensor attr    : {}".format(tensor_attr))
        #     print("tensor shape   : {}".format(timvx_tensor_shape))
        #     print("tensor qnt info: {}".format(quant_info))
        #     print("tensor data    : {}".format(np_data))
        alias_name = tflite_tensor["name"]
        return engine.create_tensor(tensor_name, tensor_dtype, tensor_attr, timvx_tensor_shape, alias_name, quant_info, np_data)


    def construct_engine_inputs(self, tflite_model_info:dict, engine:Engine, log_flag:bool=False):
        inputs = tflite_model_info["inputs"]
        tensors_info = tflite_model_info["tensors"]
        for index in range(len(inputs)):
            tensor_index = inputs[index]
            tensor_name = tensors_info[tensor_index]["name"]
            self.create_tensor(engine, tensor_name, "INPUT", tensors_info[tensor_index], log_flag)
            tensors_info[tensor_index]["visited"] = True


    def construct_engine_outputs(self, tflite_model_info:dict, engine:Engine, log_flag:bool=False):
        outputs = tflite_model_info["outputs"]
        tensors_info = tflite_model_info["tensors"]
        for index in range(len(outputs)):
            tensor_index = outputs[index]
            tensor_name = tensors_info[tensor_index]["name"]
            self.create_tensor(engine, tensor_name, "OUTPUT", tensors_info[tensor_index], log_flag)
            tensors_info[tensor_index]["visited"] = True


    def construct_engine_tensor(self, engine, tflite_model_info, tensor_index, log_flag):
        tensors_info = tflite_model_info["tensors"]
        tensor_info = tensors_info[tensor_index]
        input_tensors = tflite_model_info["inputs"]
        output_tensors = tflite_model_info["outputs"]
        if "visited" not in tensor_info.keys() or tensor_info["visited"] == False:
            tensor_name = tensor_info["name"]
            tensor_attr = ""
            if tensor_index in input_tensors:
                tensor_attr = "INPUT"
            elif tensor_index in output_tensors:
                tensor_attr = "OUTPUT"
            elif isinstance(tensor_info["buffer"], np.ndarray):
                tensor_attr = "CONSTANT"
            else:
                tensor_attr = "TRANSIENT"
            self.create_tensor(engine, tensor_name, tensor_attr, tensor_info, log_flag)
            tensors_info[tensor_index]["visited"] = True


    def construct_engine_nodes(self, tflite_model_info:dict, engine:Engine, log_flag=False):
        nodes_info = tflite_model_info["nodes"]
        for node_index in range(len(nodes_info)):
            self.construct_node(tflite_model_info, node_index, engine, log_flag)


    def convert_to_timvx(self, tflite_file:str, log_flag:bool=False):
        assert os.path.isfile(tflite_file), "{} not a valid file path!"
        with open(tflite_file, "rb") as f:
            tflite_model_data = f.read()
        engine = Engine("timvx_engine")
        assert engine.create_graph(), "timvx engine create graph fail!"
        tflite_model_info = parse_tflite_model(tflite_model_data)
        # self.construct_engine_inputs(tflite_model_info, engine, log_flag)
        # self.construct_engine_outputs(tflite_model_info, engine, log_flag)
        self.construct_engine_nodes(tflite_model_info, engine, log_flag)
        return engine