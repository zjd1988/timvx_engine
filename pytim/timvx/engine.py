# -*- coding: utf-8 -*-
import copy
import json
import numpy as np
from .lib.pytimvx import *
from .common import TimVxDataType
from .logging import get_logger

engine_logger = get_logger(__name__)

class Engine():
    def __init__(self, name:str):
        self.engine = TimVXEngine(name)
        self.mean_value = {}
        self.std_value = {}
        self.reorder = {}
        self.input_names = []    # store input names order
        self.output_names = []   # store output names order
        self.inputs_info = {}
        self.outputs_info = {}
        self.nodes_info = []
        self.tensors_info = []
        self.inputs_alias = {}
        self.outputs_alias = {}


    def set_mean_value(self, input_name:str, mean_value:list)->None:
        self.mean_value[input_name] = mean_value


    def set_std_value(self, input_name:str, std_value:list)->None:
        self.std_value[input_name] = std_value


    def set_reorder(self, input_name:str, reorder:list)->None:
        if reorder != [0, 1, 2] and reorder != [2, 1, 0]:
            assert False, "invaid channel reorder {}".format(reorder)
        self.reorder[input_name] = reorder


    def add_inputs_info(self, tensor_info:dict)->None:
        input_name = tensor_info["name"]
        assert input_name not in self.inputs_info, "tensor {} already exists!".format(input_name)
        self.inputs_info[input_name] = tensor_info
        self.input_names.append(input_name)
        if "alias" in tensor_info.keys():
            alias_name = tensor_info["alias"]
            self.inputs_alias[alias_name] = input_name
        engine_logger.info("add input tensor {}:\n{}".format(input_name, tensor_info))


    def add_outputs_info(self, tensor_info:dict)->None:
        output_name = tensor_info["name"]
        assert output_name not in self.outputs_info, "tensor {} already exists!".format(output_name)
        self.outputs_info[output_name] = tensor_info
        self.output_names.append(output_name)
        if "alias" in tensor_info.keys():
            alias_name = tensor_info["alias"]
            self.outputs_alias[alias_name] = output_name
        engine_logger.info("add output tensor {}:\n{}".format(output_name, tensor_info))


    def add_nodes_info(self, node_info:dict)->None:
        self.nodes_info.append(node_info)


    def add_tensors_info(self, tensor_info:dict)->None:
        tensor_name = tensor_info["name"]
        self.tensors_info.append(tensor_info)
        engine_logger.info("add tensor {}:\n{}".format(tensor_name, tensor_info))


    def convert_np_dtype_to_tim_dtype(self, datatype)->str:
        if datatype == np.int8:
            return "INT8"
        elif datatype == np.uint8:
            return "UINT8"
        elif datatype == np.int16:
            return "INT16"
        elif datatype == np.uint16:
            return "UINT16"
        elif datatype == np.int32:
            return "INT32"
        elif datatype == np.uint32:
            return "UINT32"
        elif datatype == np.float16:
            return "FLOAT16"
        elif datatype == np.float32:
            return "FLOAT32"
        elif datatype == np.bool:
            return "BOOL8"
        else:
            assert False, "unspoorted datatype {}, when convert np type to tim type".format(datatype)


    def convert_tim_dtype_to_np_dtype(self, datatype:str)->type:
        if datatype == "INT8":
            return np.int8
        elif datatype == "UINT8":
            return np.uint8
        elif datatype == "INT16":
            return np.int16
        elif datatype == "UINT16":
            return np.uint16
        elif datatype == "INT32":
            return np.int32
        elif datatype == "UINT32":
            return np.uint32
        elif datatype == "FLOAT16":
            return np.float16
        elif datatype == "FLOAT32":
            return np.float32
        elif datatype == "BOOL8":
            return bool
        else:
            assert False, "unspoorted datatype {}, when convert tim tensor type to np type".format(datatype)


    def get_graph_name(self)->str:
        return get_graph_name(self.engine)


    def get_tensor_size(self, tensor_name:str)->int:
        return get_tensor_size(self.engine, tensor_name)


    def create_tensor(self, tensor_name:str, tensor_dtype:str, tensor_attr:str, \
        tensor_shape:list, alias_name:str="", quant_info:dict={}, np_data:np.array=np.array([]))->dict:

        assert tensor_dtype in TimVxDataType, "tim-vx not support {} datatype".format(tensor_dtype)
        tensor_info = {}
        tensor_info["shape"] = tensor_shape
        tensor_info["data_type"] = tensor_dtype
        tensor_info["attribute"] = tensor_attr
        if len(quant_info.keys()) != 0:
            tensor_info["quant_info"] = quant_info
        assert create_tensor(self.engine, tensor_name, tensor_info, np_data), "creat tensor {} fail!".format(tensor_name)
        # add engine tensor_info stat
        tensor_stat_info = {}
        tensor_stat_info["name"] = tensor_name
        tensor_stat_info["data_type"] = self.convert_tim_dtype_to_np_dtype(tensor_dtype)
        tensor_stat_info["attribute"] = tensor_attr
        tensor_stat_info["shape"] = tensor_shape
        tensor_stat_info["quant_info"] = quant_info
        if np_data.size != 0:
            tensor_stat_info["data"] = np_data
        if "alias" in tensor_info.keys():
            tensor_stat_info["alias"] = tensor_info["alias"]
        if tensor_attr == "INPUT":
            self.add_inputs_info(tensor_stat_info)
        elif tensor_attr == "OUTPUT":
            self.add_outputs_info(tensor_stat_info)
        else:
            self.add_tensors_info(tensor_stat_info)
        return tensor_stat_info


    def copy_data_from_tensor(self, tensor_name:str, np_data:np.array)->bool:
        return copy_data_from_tensor(self.engine, tensor_name, np_data)


    def copy_data_to_tensor(self, tensor_name:str, np_data:np.array)->bool:
        return copy_data_to_tensor(self.engine, tensor_name, np_data)


    def create_operation(self, op_info:dict)->bool:
        ret = create_operation(self.engine, op_info)
        op_name = op_info["op_name"]
        if ret and "op_inputs" in op_info.keys():
            op_inputs = op_info["op_inputs"]
            ret = bind_inputs(self.engine, op_name, op_inputs)
        if ret and "op_outputs" in op_info.keys():
            op_outputs = op_info["op_outputs"]
            ret = bind_outputs(self.engine, op_name, op_outputs)
        self.add_nodes_info(op_info)
        return ret


    def get_op_info(self, op_name:str)->dict:
        return get_op_info(self.engine, op_name)


    def bind_input(self, op_name:str, tensor_name:str)->bool:
        return bind_input(self.engine, op_name, tensor_name)


    def bind_output(self, op_name:str, tensor_name:str)->bool:
        return bind_output(self.engine, op_name, tensor_name)


    def bind_inputs(self, op_name:str, tensor_names:list)->bool:
        return bind_inputs(self.engine, op_name, tensor_names)


    def bind_outputs(self, op_name:str, tensor_names:list)->bool:
        return bind_outputs(self.engine, op_name, tensor_names)


    def create_graph(self)->bool:
        return create_graph(self.engine)


    def verify_graph(self)->bool:
        return verify_graph(self.engine)


    def compile_graph(self)->bool:
        return compile_graph(self.engine)


    def compile_to_binary(self)->bytearray:
        return compile_to_binary(self.engine)


    def run_graph(self, input_dict:dict, output_name_list:list=[], pass_through:bool=True, want_float:bool=False)->list:
        for input_name in input_dict.keys():
            # check input tensor name valid
            assert input_name in self.inputs_info.keys() or input_name in self.inputs_alias.keys(), \
                "invalid input tensor name {}".format(input_name)

            # get real input name if contain alias
            real_input_name = input_name
            if input_name in self.inputs_alias.keys():
                real_input_name = self.inputs_alias[input_name]

            # get input tensor infos
            tensor_info = self.inputs_info[real_input_name]

            # check input data type
            input_data = input_dict[input_name]
            assert type(input_data) == np.ndarray, "input tensor {} data only support numpy array".format(input_name)

            # check input data shape
            # timvx tensor dims is reverse order with np dims
            np_shape = input_data.shape
            tensor_dtype = tensor_info["data_type"]
            tensor_shape = copy.deepcopy(tensor_info["shape"])
            tensor_shape.reverse()

            # set input tensor data
            if pass_through:
                assert tensor_shape != np_shape, \
                    "input tensor {} numpy shape:{} not equal to tensor shape:{}".format(input_name, np_shape, tensor_shape)
                engine_input = input_data
            else:
                mean_value = [0.0, ]
                std_value = [1.0, ]
                if real_input_name in self.mean_value.keys():
                    mean_value = self.mean_value[real_input_name]
                if real_input_name in self.std_value.keys():
                    std_value = self.std_value[real_input_name]
                engine_input = (input_data.astype(np.float32) - mean_value) / std_value
                if self.reorder == [2, 1, 0]:
                    assert len(np_shape) == 3, "need a hwc format input, please check!"
                    h,w,c = np_shape
                    assert c == 3 or c == 1, "input channel should be 1 or 3"
                    engine_input = engine_input[:,:,::-1]
                    engine_input = engine_input.transpose((2, 0, 1))

                scale = 1.0
                zero_point = 0.0
                if "scale" in tensor_info["quant_info"]:
                    scale = tensor_info["quant_info"]["scale"]
                if "zero_point" in tensor_info["quant_info"]:
                    zero_point = tensor_info["quant_info"]["zero_point"]
                engine_input = (engine_input / scale + zero_point).reshape(tensor_shape).astype(tensor_dtype)
            assert self.copy_data_to_tensor(input_name, engine_input), "set input tensor {} fail!".format(input_name)

        # run graph
        assert run_graph(self.engine), "run graph fail!"

        # get output tensors
        outputs = []
        if output_name_list == []:
            if self.outputs_alias != {}:
                output_name_list = list(self.outputs_alias.keys())
            else:
                output_name_list = copy.deepcopy(self.output_names)
        for output_index in range(len(output_name_list)):
            output_name = output_name_list[output_index]
            real_output_name = output_name
            if output_name in self.outputs_alias.keys():
                real_output_name = self.outputs_alias[output_name]
            assert real_output_name in self.outputs_info.keys(), "invalid output tensor name {}".format(output_name)

            # get input tensor infos
            tensor_info = self.outputs_info[real_output_name]

            # timvx tensor dims is reverse order with np dims
            tensor_shape = copy.deepcopy(tensor_info["shape"])
            tensor_shape.reverse()
            tensor_dtype = tensor_info["data_type"]
            output_data = np.zeros(tensor_shape).astype(tensor_dtype)
            assert self.copy_data_from_tensor(output_name, output_data), "get output tensor {} fail!".format(output_name)

            if want_float:
                scale = 1.0
                zero_point = 0.0
                if "scale" in tensor_info["quant_info"]:
                    scale = tensor_info["quant_info"]["scale"]
                if "zero_point" in tensor_info["quant_info"]:
                    zero_point = tensor_info["quant_info"]["zero_point"]
                output_data = output_data.astype(np.float32)
                output_data = (output_data - zero_point) * scale
            outputs.append(output_data)
        return outputs


    def export_graph(self, graph_json_file:str, weight_bin_file:str, log_flag:bool=False)->bool:
        graph_json_dict = {}
        # init norm_info
        engine_logger.info("prepare norm ...")
        norm_info = {}
        norm_info["mean"] = self.mean_value
        norm_info["std"] = self.std_value
        norm_info["reorder"] = self.reorder
        graph_json_dict["norm"] = norm_info
        if log_flag:
            engine_logger.info(graph_json_dict["norm"])

        # init inputs_info
        engine_logger.info("prepare inputs ...")
        inputs_info = []
        for input_name in self.inputs_info.keys():
            input_tensor = {}
            tensor_info = self.inputs_info[input_name]
            for item_key in tensor_info.keys():
                item_value = tensor_info[item_key]
                if item_key == "data_type":
                    item_value = self.convert_np_dtype_to_tim_dtype(item_value)
                input_tensor[item_key] = item_value
            if log_flag:
                engine_logger.info(input_tensor)
            inputs_info.append(input_tensor)
        graph_json_dict["inputs"] = inputs_info

        # init outputs_info
        engine_logger.info("prepare outputs ...")
        outputs_info = []
        for output_name in self.outputs_info.keys():
            output_tensor = {}
            tensor_info = self.outputs_info[output_name]
            for item_key in tensor_info.keys():
                item_value = tensor_info[item_key]
                if item_key == "data_type":
                    item_value = self.convert_np_dtype_to_tim_dtype(item_value)
                output_tensor[item_key] = item_value
            if log_flag:
                engine_logger.info(output_tensor)
            outputs_info.append(output_tensor)
        graph_json_dict["outputs"] = outputs_info

        # init tensors_info
        engine_logger.info("prepare tensors ...")
        weight_offset = 0
        weight_bin_list = []
        tensors_info = []
        for index in range(len(self.tensors_info)):
            new_tensor_info = {}
            tensor_info = self.tensors_info[index]
            for item_key in tensor_info.keys():
                item_value = tensor_info[item_key]
                if item_key == "data_type":
                    item_value = self.convert_np_dtype_to_tim_dtype(item_value)
                    new_tensor_info[item_key] = item_value
                elif item_key == "data":
                    length_item_key = "length"
                    length_item_value = len(tensor_info[item_key].tobytes())
                    offset_item_key = "offset"
                    offset_item_value = weight_offset
                    weight_offset += len(tensor_info[item_key].tobytes())
                    weight_bin_list.append(tensor_info[item_key].tobytes())
                    new_tensor_info[length_item_key] = length_item_value
                    new_tensor_info[offset_item_key] = offset_item_value
                else:
                    new_tensor_info[item_key] = item_value
            if log_flag:
                engine_logger.info(new_tensor_info)
            tensors_info.append(new_tensor_info)
        graph_json_dict["tensors"] = tensors_info

        # init nodes_info/inputs_alias/outputs_alias
        engine_logger.info("prepare nodes/inputs_alias/outputs_alias ...")
        graph_json_dict["nodes"] = self.nodes_info
        graph_json_dict["inputs_alias"] = self.inputs_alias
        graph_json_dict["outputs_alias"] = self.outputs_alias
        if log_flag:
            engine_logger.info(graph_json_dict["nodes"])
            engine_logger.info(graph_json_dict["inputs_alias"])
            engine_logger.info(graph_json_dict["outputs_alias"])
        # dump to json file/bin file
        engine_logger.info("write to file ...")
        graph_json_obj = json.dumps(graph_json_dict, indent=4)
        with open(graph_json_file, "w") as f:
            f.write(graph_json_obj)

        with open(weight_bin_file, "wb") as f:
            for index in range(len(weight_bin_list)):
                f.write(weight_bin_list[index])

        engine_logger.info("export success.")
        return True