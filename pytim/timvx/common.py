# -*- coding: utf-8 -*-
import math
import numpy as np
from .lib.pytimvx import *
PadType = ["NONE", "AUTO", "VALID", "SAME"]
PoolType = ["MAX", "AVG", "L2", "AVG_ANDROID"]
RoundType = ["CEILING", "FLOOR"]
OverflowPolicy = ["WRAP", "SATURATE"]
RoundingPolicy = ["TO_ZERO", "RTNE"]
ResizeType = ["NEAREST_NEIGHBOR", "BILINEAR", "AREA"]
DataLayout = [ "ANY", "WHCN", "CWHN", "IcWHOc", "OcIcWH", "IcOcWH", "WHIcOc", "WCN", "WIcOc"]
TimVxDataType = ["INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "FLOAT16", "FLOAT32", "BOOL8"]
QuantType = ["NONE", "ASYMMETRIC", "SYMMETRIC_PER_CHANNEL"]


def setLogLevel(log_level:str="DEBUG"):
    LOG_LEVEL_MAP = {"TRACE" : 0,
                     "DEBUG" : 1,
                     "INFO"  : 2,
                     "WARN"  : 3,
                     "ERROR" : 4}
    return set_log_level(LOG_LEVEL_MAP[log_level])


def quantizationParams(f_min:float, f_max:float, data_type:type):
    type_info = np.iinfo(data_type)
    zero_point = 0
    scale = 0
    qmin = type_info.min
    qmax = type_info.max
    qmin_double = float(qmin)
    qmax_double = float(qmax)
    #   // 0 should always be a representable value. Let's assume that the initial
    #   // min,max range contains 0.
    if f_min == f_max:
        # // Special case where the min,max range is a point. Should be {0}.
        return scale, zero_point


    #   // General case.
    #   //
    #   // First determine the scale.
    scale = (f_max - f_min) / (qmax_double - qmin_double)

    #   // Zero-point computation.
    #   // First the initial floating-point computation. The zero-point can be
    #   // determined from solving an affine equation for any known pair
    #   // (real value, corresponding quantized value).
    #   // We know two such pairs: (rmin, qmin) and (rmax, qmax).
    #   // The arithmetic error on the zero point computed from either pair
    #   // will be roughly machine_epsilon * (sum of absolute values of terms)
    #   // so we want to use the variant that adds the smaller terms.
    zero_point_from_min = qmin_double - f_min / scale
    zero_point_from_max = qmax_double - f_max / scale

    zero_point_from_min_error = abs(qmin_double) + abs(f_min / scale)

    zero_point_from_max_error = abs(qmax_double) + abs(f_max / scale)

    zero_point_double = zero_point_from_min if zero_point_from_min_error < zero_point_from_max_error else zero_point_from_max

    #   // Now we need to nudge the zero point to be an integer
    #   // (our zero points are integer, and this is motivated by the requirement
    #   // to be able to represent the real value "0" exactly as a quantized value,
    #   // which is required in multiple places, for example in Im2col with SAME
    #   //  padding).

    nudged_zero_point = 0
    if zero_point_double < qmin_double:
        nudged_zero_point = qmin
    elif zero_point_double > qmax_double:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = round(zero_point_double)

    #   // The zero point should always be in the range of quantized value,
    #   // // [qmin, qmax].

    zero_point = nudged_zero_point
    #   // finally, return the values
    return scale, zero_point


def quantize(data:'list|np.array', scale:float, zero_point:int, dest_type:type)->list:
    type_info = np.iinfo(dest_type)
    min_value = type_info.min
    max_value = type_info.max
    if list == type(data):
        np_array = np.array(data)
    else:
        np_array = data
    np_array_q = np.round((np_array / scale) + zero_point)
    np_array_q[np_array_q > max_value] = max_value
    np_array_q[np_array_q < min_value] = min_value
    return np_array_q.astype(dest_type)


def dequantize(data:'list|np.array', scale:float, zero_point:int)->np.array:
    if list == type(data):
        np_array = np.array(data)
    else:
        np_array = data
    np_array = np_array.astype(np.float32)
    return ((np_array - zero_point) * scale).astype(np.float32)


class Quantization():
    def __init__(self, scale:'int|list', zp:'int|list', quant_type:str="NONE", channel_dim:int=-1):
        self.type = quant_type
        self.channel_dim = channel_dim
        self.scales = list(scale)
        self.zero_points = list(zp)


    def type(self)->str:
        return type


    def setType(self, type:str)->None:
        if type not in QuantType:
            print("")
        else:
            self.type = type
    

    def channelDim(self)->int:
        return self.channel_dim

    
    def setChannelDim(self, channel_dim:int)->None:
        self.channel_dim = channel_dim


    def scales(self)->list:
        return self.scales

    
    def setScales(self, scales:list)->None:
        self.scales = scales


    def zeroPoints(self)->list:
        return self.zero_points

    
    def setZeroPoints(self, zps:list)->None:
        self.zero_points = zps


def ConstructActivationOpConfig(op_name:str, activation_type:str, parameter:dict={}, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    # 1 prelu parameter
    # axis = None
    # 2 leakyrelu parameter
    # ratio = None
    # 3 linear parameter
    # a = None b = 0.0
    # 4 gelu parameter
    # approximate = True 
    valid_act_type = ["Relu", "Relu1", "Relu6", "Elu", "Sigmoid", "Mish", "HardSigmoid",
        "SoftRelu", "HardSwish", "Swish", "Prelu", "Tanh", "LeakyRelu", "Linear", "Gelu"]
    assert activation_type in valid_act_type, "activation_type:{} is not in {}".format(activation_type, valid_act_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Activation"
    op_attr = {}
    op_attr["activation_type"] = activation_type
    op_attr.update(parameter)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructAddNOpConfig(op_name:str, num_inputs:int={}, op_inputs:list=[], op_outputs:list=[])->dict:

    assert num_inputs >= 0, "num_inputs: {} must greater than 0".format(num_inputs)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "AddN"
    op_attr = {}
    op_attr["num_inputs"] = num_inputs
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructArgOpConfig(op_name:str, arg_type:str, parameter:dict={}, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    # 1 Max parameter
    # axis = None
    # 2 Min parameter
    # axis = None
    valid_arg_type = ["Max", "Min"]
    assert arg_type in valid_arg_type, "arg_type:{} is not in {}".format(arg_type, valid_arg_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Arg"
    op_attr = {}
    op_attr["arg_type"] = arg_type
    op_attr.update(parameter)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructBatch2SpaceOpConfig(op_name:str, block_size:list, crop:list=[], layout:str="WHCN",
    op_inputs:list=[], op_outputs:list=[])->dict:
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Batch2Space"
    op_attr = {}
    op_attr["block_size"] = block_size
    op_attr["crop"] = crop
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructBatchNormOpConfig(op_name:str, eps:float, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "BatchNorm"
    op_attr = {}
    op_attr["eps"] = eps
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructClipOpConfig(op_name:str, min:float, max:float, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Clip"
    op_attr = {}
    op_attr["min"] = min
    op_attr["max"] = max
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructConcatOpConfig(op_name:str, axis:int, op_inputs:list=[], op_outputs:list=[])->dict:

    assert axis >= 0, "axis:{} should greater than 0".format(axis)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Concat"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["input_cnt"] = len(op_inputs)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructConv1dOpConfig(op_name:str, stride:int, dilation:int, ksize:int=0, padding:str="AUTO", 
    pad:list=[0, 0], weights:int=0, multiplier:int=0, kernel_layout:str="WHIcOc", 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Conv1d"
    op_attr = {}
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["dilation"] = dilation
    op_attr["padding"] = padding
    op_attr["pad"] = pad
    op_attr["weights"] = weights
    op_attr["multiplier"] = multiplier
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructConv2dOpConfig(op_name:str, stride:list, dilation:list, ksize:list=[0, 0], padding:str="AUTO", 
    pad:list=[0, 0, 0, 0], weights:int=0, multiplier:int=0, input_layout:str="WHCN", 
    kernel_layout:str="WHIcOc", op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert input_layout in DataLayout, "input_layout:{} is not in {}".format(input_layout, DataLayout)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Conv2d"
    op_attr = {}
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["dilation"] = dilation
    op_attr["padding"] = padding
    op_attr["pad"] = pad
    op_attr["weights"] = weights
    op_attr["multiplier"] = multiplier
    op_attr["input_layout"] = input_layout
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructDeConv1dOpConfig(op_name:str, stride:int, output_padding:int, pad_type:str="AUTO", 
    oc_count:int=0, ksize:int=0, pad:list=[0, 0], group:int=1, kernel_layout:str="WHIcOc", 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert pad_type in PadType, "pad_type:{} is not in {}".format(pad_type, PadType)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "DeConv1d"
    op_attr = {}
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["output_padding"] = output_padding
    op_attr["pad_type"] = pad_type
    op_attr["pad"] = pad
    op_attr["group"] = group
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructDeConv2dOpConfig(op_name:str, oc_count:int, stride:list, ksize:list, 
    output_padding:list, pad_type:str="AUTO", pad:list=[0, 0, 0, 0], group:int=1, 
    kernel_layout:str="WHIcOc", op_inputs:list=[], op_outputs:list=[])->dict:

    assert pad_type in PadType, "pad_type:{} is not in {}".format(pad_type, PadType)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "DeConv2d"
    op_attr = {}
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["oc_count"] = oc_count
    op_attr["pad_type"] = pad_type
    op_attr["pad"] = pad
    op_attr["output_padding"] = output_padding
    op_attr["group"] = group
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructDepth2SpaceOpConfig(op_name:str, block_size:int, layout:str="WHCN", 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Depth2Space"
    op_attr = {}
    op_attr["block_size"] = block_size
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructDropoutOpConfig(op_name:str, ratio:float, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Dropout"
    op_attr = {}
    op_attr["ratio"] = ratio
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructEltwiseOpConfig(op_name:str, eltwise_type:str, parameter:dict={}, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    # Multiply/Div parameter
    # scale = 1.0
    valid_eltwise_type = ["Minimum", "Maximum", "Add", "Sub", "Pow", "FloorDiv", "Multiply", "Div"]
    assert eltwise_type in valid_eltwise_type, "eltwise_type:{} is not in {}".format(eltwise_type, valid_eltwise_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Eltwise"
    op_attr = {}
    op_attr["eltwise_type"] = eltwise_type
    op_attr.update(parameter)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    
    return op_info_dict


def ConstructFullyConnectedOpConfig(op_name:str, axis:int, weights:int=0, op_inputs:list=[], op_outputs:list=[])->dict:

    assert axis >= 0, "axis:{} should >= 0".format(axis)
    assert weights >= 0, "weights:{} should >= 0".format(weights)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "FullyConnected"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["weights"] = weights
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructGatherOpConfig(op_name:str, axis:int, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Gather"
    op_attr = {}
    op_attr["axis"] = axis
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructGatherNdOpConfig(op_name:str, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "GatherNd"
    op_attr = {}
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructGroupedConv2dOpConfig(op_name:str, stride:list, dilation:list, grouped_number:int, padding:str="AUTO", 
    pad:list=[0, 0, 0, 0], input_layout:str="WHCN", kernel_layout:str="WHIcOc", 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert input_layout in DataLayout, "input_layout:{} is not in {}".format(input_layout, DataLayout)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "GroupedConv2d"
    op_attr = {}
    op_attr["stride"] = stride
    op_attr["dilation"] = dilation
    op_attr["grouped_number"] = grouped_number
    op_attr["padding"] = padding
    op_attr["pad"] = pad
    op_attr["input_layout"] = input_layout
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructInstanceNormalizationOpConfig(op_name:str, eps:float=1e-5, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "InstanceNormalization"
    op_attr = {}
    op_attr["eps"] = eps
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructL2NormalizationOpConfig(op_name:str, axis:int, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "L2Normalization"
    op_attr = {}
    op_attr["axis"] = axis
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructLayerNormalizationOpConfig(op_name:str, axis:int=0, eps:float=1e-5, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "LayerNormalization"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["eps"] = eps
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructLocalResponseNormalizationOpConfig(op_name:str, size:int, alpha:float, beta:float,
    bias:float, axis:int, op_inputs:list=[], op_outputs:list=[])->dict:

    assert size >= 0, "size:{} must greater than 0".format(size)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "LocalResponseNormalization"
    op_attr = {}
    op_attr["size"] = size
    op_attr["alpha"] = alpha
    op_attr["beta"] = beta
    op_attr["bias"] = bias
    op_attr["axis"] = axis
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructLogicalOpConfig(op_name:str, logical_type:str, parameter:dict={}, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    # Add/Or parameter
    valid_logical_type = ["Add", "Or"]
    assert logical_type in valid_logical_type, "logical_type:{} is not in {}".format(logical_type, valid_logical_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Logical"
    op_attr = {}
    op_attr["logical_type"] = logical_type
    op_attr.update(parameter)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    
    return op_info_dict


def ConstructLogSoftmaxOpConfig(op_name:str, axis:int, beta:float=1.0,  
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "LogSoftmax"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["beta"] = beta
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    
    return op_info_dict


def ConstructMatmulOpConfig(op_name:str, transpose_a:bool=False, transpose_b:bool=False, 
    adjoint_a:bool=False, adjoint_b:bool=False, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Matmul"
    op_attr = {}
    op_attr["transpose_a"] = transpose_a
    op_attr["transpose_b"] = transpose_b
    op_attr["adjoint_a"] = adjoint_a
    op_attr["adjoint_b"] = adjoint_b
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructMaxpoolWithArgmaxOpConfig(op_name:str, padding:str, ksize:list, 
    stride:list, round_type:str="FLOOR", layout:str="WHCN", 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert round_type in RoundType, "round_type:{} is not in {}".format(round_type, RoundType)
    assert layout in DataLayout, "layout:{} is not in {}".format(layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "MaxpoolWithArgmax"
    op_attr = {}
    op_attr["padding"] = padding
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["round_type"] = round_type
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructMaxUnpool2dOpConfig(op_name:str, ksize:list, stride:list, layout:str="WHCN", 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert layout in DataLayout, "layout:{} is not in {}".format(layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "MaxUnpool2d"
    op_attr = {}
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructMomentsOpConfig(op_name:str, axes:list, keep_dims:bool=False, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Moments"
    op_attr = {}
    op_attr["axes"] = axes
    op_attr["keep_dims"] = keep_dims
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructNBGOpConfig(op_name:str, binary:int, input_count:int, output_count:int, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "NBG"
    op_attr = {}
    op_attr["binary"] = binary
    op_attr["input_count"] = input_count
    op_attr["output_count"] = output_count
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructPadOpConfig(op_name:str, front_size:list, back_size:list, const_val:int, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Pad"
    op_attr = {}
    op_attr["front_size"] = front_size
    op_attr["back_size"] = back_size
    op_attr["const_val"] = const_val
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructPool2dOpConfig(op_name:str, type:str, ksize:list=[], stride:list=[], padding:str="AUTO",
    pad:list=[0, 0, 0, 0], input_size:list=[], output_size:list=[], round_type:str="FLOOR", 
    layout:str="WHCN", op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert round_type in RoundType, "round_type:{} is not in {}".format(round_type, RoundType)
    assert layout in DataLayout, "layout:{} is not in {}".format(layout, DataLayout)
    if len(input_size) == 0:
        assert len(ksize) and len(stride), "ksize and stride len should > 0, when input_size len is 0"
    if len(input_size) > 0:
        assert len(ksize) == 0 and len(stride) == 0, "ksize and stride len should be 0, when input_size len > 0"
    if padding != "AUTO":
        assert pad == [0, 0, 0, 0], "pad should be [0, 0, 0, 0], when padding is not AUTO"
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Pool2d"
    op_attr = {}
    op_attr["type"] = type
    if len(input_size) > 0 and len(output_size) == 0:
        op_attr["input_size"] = input_size
    elif len(input_size) > 0 and len(output_size) > 0:
        op_attr["input_size"] = input_size
        op_attr["output_size"] = output_size
    elif len(input_size) == 0 and padding == "AUTO":
        op_attr["pad"] = pad
        op_attr["ksize"] = ksize
        op_attr["stride"] = stride
    else:
        op_attr["padding"] = padding
        op_attr["ksize"] = ksize
        op_attr["stride"] = stride
    op_attr["round_type"] = round_type
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructReduceOpConfig(op_name:str, axis:list, keep_dims:bool, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Reduce"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["keep_dims"] = keep_dims
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructReorgOpConfig(op_name:str, stride:int, op_inputs:list=[], op_outputs:list=[])->dict:

    assert stride >= 0, "stride: {} must greater than 0".format(stride)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Reorg"
    op_attr = {}
    op_attr["stride"] = stride
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructRelationalOperationsOpConfig(op_name:str, relational_type:str, op_inputs:list=[], op_outputs:list=[])->dict:

    # Greater/GreaterOrEqual/Less/LessOrEqual/NotEqual/Equal 
    valid_relational_type = ["Greater", "GreaterOrEqual", "Less", "LessOrEqual", "NotEqual", "Equal", ]
    assert relational_type in valid_relational_type, \
        "relationa_type:{} is not in {}".format(relational_type, valid_relational_type)

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "RelationalOperations"
    op_attr = {}
    op_attr["relational_type"] = relational_type
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructReshapeOpConfig(op_name:str, size:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Reshape"
    op_attr = {}
    op_attr["size"] = size
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructResizeOpConfig(op_name:str, type:str, factor:float, align_corners:bool,
        half_pixel_centers:bool, target_height:int, target_width:int, 
        layout:str="WHCN", op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Resize"
    op_attr = {}
    op_attr["type"] = type
    op_attr["factor"] = factor
    op_attr["align_corners"] = align_corners
    op_attr["half_pixel_centers"] = half_pixel_centers
    op_attr["target_height"] = target_height
    op_attr["target_width"] = target_width
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructResize1dOpConfig(op_name:str, type:str, factor:float, align_corners:bool,
        half_pixel_centers:bool, target_size:int, layout:str="WHCN", 
        op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Resize1d"
    op_attr = {}
    op_attr["type"] = type
    op_attr["factor"] = factor
    op_attr["align_corners"] = align_corners
    op_attr["half_pixel_centers"] = half_pixel_centers
    op_attr["target_size"] = target_size
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructReverseOpConfig(op_name:str, axis:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Reverse"
    op_attr = {}
    op_attr["axis"] = axis
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructScatterNDOpConfig(op_name:str, shape:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "ScatterND"
    op_attr = {}
    op_attr["shape"] = shape
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSelectOpConfig(op_name:str, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Select"
    op_attr = {}
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSimpleOperationsOpConfig(op_name:str, simple_type:str, op_inputs:list=[], op_outputs:list=[])->dict:

    # DataConvert/Neg/Abs/Sin/Exp/Log/Sqrt/Rsqrt/Square/LogicalNot/Floor/Cast 
    valid_simple_type = ["DataConvert", "Neg", "Abs", "Sin", "Exp", 
        "Log", "Sqrt", "Rsqrt", "Square", "LogicalNot", "Floor", "Cast"]
    assert simple_type in valid_simple_type, \
        "simple_type:{} is not in {}".format(simple_type, valid_simple_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "SimpleOperations"
    op_attr = {}
    op_attr["simple_type"] = simple_type
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSliceOpConfig(op_name:str, dims:int, start:list, length:list, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert dims >= 0, "dims: {} must greater than 0".format(dims)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Slice"
    op_attr = {}
    op_attr["dims"] = dims
    op_attr["start"] = start
    op_attr["length"] = length
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSoftmaxOpConfig(op_name:str, beta:float, axis:int, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Softmax"
    op_attr = {}
    op_attr["beta"] = beta
    op_attr["axis"] = axis
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSpace2BatchOpConfig(op_name:str, block_size:list, pad:list, layout:str="WHCN",  
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Space2Batch"
    op_attr = {}
    op_attr["block_size"] = block_size
    op_attr["pad"] = pad
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSpace2DepthOpConfig(op_name:str, block_size:list, layout:str="WHCN",  
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Space2Depth"
    op_attr = {}
    op_attr["block_size"] = block_size
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSpatialTransformerOpConfig(op_name:str, output_h:int, output_w:int, 
    has_theta_1_1:bool, has_theta_1_2:bool, has_theta_1_3:bool, 
    has_theta_2_1:bool, has_theta_2_2:bool, has_theta_2_3:bool, 
    theta_1_1:float, theta_1_2:float, theta_1_3:float, 
    theta_2_1:float, theta_2_2:float, theta_2_3:float, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert output_h >= 0, "output_h: {} must greater than 0".format(output_h)
    assert output_w >= 0, "output_w: {} must greater than 0".format(output_w)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "SpatialTransformer"
    op_attr = {}
    op_attr["output_h"] = output_h
    op_attr["output_w"] = output_w
    op_attr["has_theta_1_1"] = has_theta_1_1
    op_attr["has_theta_1_2"] = has_theta_1_2
    op_attr["has_theta_1_3"] = has_theta_1_3
    op_attr["has_theta_2_1"] = has_theta_2_1
    op_attr["has_theta_2_2"] = has_theta_2_2
    op_attr["has_theta_2_3"] = has_theta_2_3
    op_attr["theta_1_1"] = theta_1_1
    op_attr["theta_1_2"] = theta_1_2
    op_attr["theta_1_3"] = theta_1_3
    op_attr["theta_2_1"] = theta_2_1
    op_attr["theta_2_2"] = theta_2_2
    op_attr["theta_2_3"] = theta_2_3
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSplitOpConfig(op_name:str, axis:int, slices:list, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert axis >= 0, "axis: {} must greater than 0".format(axis)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Split"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["slices"] = slices
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSqueezeOpConfig(op_name:str, axis:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Squeeze"
    op_attr = {}
    op_attr["axis"] = axis
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructStackOpConfig(op_name:str, axis:int, input_cnt:int, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert axis >= 0, "axis: {} must greater than 0".format(axis)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Stack"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["input_cnt"] = input_cnt
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructStridedSliceOpConfig(op_name:str, begin_dims:list, end_dims:list, stride_dims:list, 
    begin_mask:int, end_mask:int, shrink_axis_mask=int, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "StridedSlice"
    op_attr = {}
    op_attr["begin_dims"] = begin_dims
    op_attr["end_dims"] = end_dims
    op_attr["stride_dims"] = stride_dims
    op_attr["begin_mask"] = begin_mask
    op_attr["end_mask"] = end_mask
    op_attr["shrink_axis_mask"] = shrink_axis_mask
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructTileOpConfig(op_name:str, multiples:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Tile"
    op_attr = {}
    op_attr["multiples"] = multiples
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructTransposeOpConfig(op_name:str, perm:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Transpose"
    op_attr = {}
    op_attr["perm"] = perm
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructUnstackOpConfig(op_name:str, axis:int, output_num:int, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert output_num >= 0, "output_num: {} must greater than 0".format(output_num)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Unstack"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["output_num"] = output_num
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict
