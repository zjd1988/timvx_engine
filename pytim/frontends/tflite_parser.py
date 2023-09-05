# -*- coding: utf-8 -*-
"""tflite frontend."""
import tflite
import numpy as np

FUSED_ACTIVATION_FUNCTION_STR_MAP = {
  0 : "NONE",
  1 : "RELU",
  2 : "RELU_N1_TO_1",
  3 : "RELU6",
  4 : "TANH",
  5 : "SIGN_BIT",
}

TFLITE_PADDING_TYPE_STR_MAP = {
    0 : "SAME",
    1 : "VALID"
}

def parse_null_options(op_opt):
    op_attr = {}
    return op_attr

def parse_conv2d_options(op_opt):
    option = tflite.Conv2DOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["padding"] = TFLITE_PADDING_TYPE_STR_MAP[option.Padding()]
    op_attr["stride_w"] = option.StrideW()
    op_attr["stride_h"] = option.StrideH()
    op_attr["dilation_w_factor"] = option.DilationWFactor()
    op_attr["dilation_h_factor"] = option.DilationHFactor()
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    return op_attr

def parse_conv3d_options(op_opt):
    option = tflite.Conv3DOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["padding"] = TFLITE_PADDING_TYPE_STR_MAP[option.Padding()]
    op_attr["stride_d"] = option.StrideD()
    op_attr["stride_w"] = option.StrideW()
    op_attr["stride_h"] = option.StrideH()
    op_attr["dilation_d_factor"] = option.DilationDFactor()
    op_attr["dilation_w_factor"] = option.DilationWFactor()
    op_attr["dilation_h_factor"] = option.DilationHFactor()
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    return op_attr

def parse_pool2d_options(op_opt):
    option = tflite.Pool2DOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["padding"] = TFLITE_PADDING_TYPE_STR_MAP[option.Padding()]
    op_attr["stride_w"] = option.StrideW()
    op_attr["stride_h"] = option.StrideH()
    op_attr["filter_width"] = option.FilterWidth()
    op_attr["filter_height"] = option.FilterHeight()
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    return op_attr

def parse_depthwiseconv2d_options(op_opt):
    option = tflite.DepthwiseConv2DOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["padding"] = TFLITE_PADDING_TYPE_STR_MAP[option.Padding()]
    op_attr["stride_w"] = option.StrideW()
    op_attr["stride_h"] = option.StrideH()
    op_attr["depth_multiplier"] = option.DepthMultiplier()
    op_attr["dilation_w_factor"] = option.DilationWFactor()
    op_attr["dilation_h_factor"] = option.DilationHFactor()
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    return op_attr

def parse_fullyconnected_options(op_opt):
    option = tflite.FullyConnectedOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    op_attr["weights_format"] = option.WeightsFormat()
    op_attr["keep_num_dims"] = option.KeepNumDims()
    op_attr["asymmetric_quantize_inputs"] = option.AsymmetricQuantizeInputs()
    return op_attr

def parse_softmax_options(op_opt):
    option = tflite.SoftmaxOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["beta"] = option.Beta()
    return op_attr

def parse_concatenation_options(op_opt):
    option = tflite.ConcatenationOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["axis"] = option.Axis()
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    return op_attr

def parse_add_options(op_opt):
    option = tflite.AddOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["pot_scale_int16"] = option.PotScaleInt16()
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    return op_attr

def parse_mul_options(op_opt):
    option = tflite.MulOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    return op_attr

def parse_resizebilinear_options(op_opt):
    option = tflite.ResizeBilinearOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["new_height"] = option.NewHeight()
    op_attr["new_width"] = option.NewWidth()
    op_attr["align_corners"] = option.AlginCorner()
    op_attr["half_pixel_centers"] = option.HalfPixelCenters()
    return op_attr

def parse_resizenearesetneighbor_options(op_opt):
    option = tflite.ResizeNearestNeighborOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["align_corners"] = option.AlignCorners()
    op_attr["half_pixel_centers"] = option.HalfPixelCenters()
    return op_attr

def parse_pad_options(op_opt):
    return parse_null_options(op_opt)

def parse_padv2_options(op_opt):
    return parse_null_options(op_opt)

def parse_reshape_options(op_opt):
    option = tflite.ReshapeOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["new_shape"] = option.NewShape()
    return op_attr

def parse_spacetodepth_options(op_opt):
    option = tflite.SpaceToDepthOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["block_size"] = option.BlockSize()
    return op_attr

def parse_depthtospace_options(op_opt):
    option = tflite.DepthToSpaceOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["block_size"] = option.BlockSize()
    return op_attr

def parse_sub_options(op_opt):
    option = tflite.SubOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["pot_scale_int16"] = option.PotSacleInt16()
    return op_attr

def parse_div_options(op_opt):
    option = tflite.DivOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["fused_activation_function"] = FUSED_ACTIVATION_FUNCTION_STR_MAP[option.FusedActivationFunction()]
    return op_attr

def parse_gather_options(op_opt):
    option = tflite.GatherOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["axis"] = option.Axis()
    op_attr["batch_dims"] = option.BatchDims()
    return op_attr

def parse_transpose_options(op_opt):
    return parse_null_options(op_opt)

def parse_exp_options(op_opt):
    return parse_null_options(op_opt)

def parse_cos_options(op_opt):
    return parse_null_options(op_opt)

def parse_reduce_options(op_opt):
    option = tflite.ReducerOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["keep_dims"] = option.Axis()
    return op_attr

def parse_squeeze_options(op_opt):
    option = tflite.SqueezeOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["squeeze_dims"] = option.SqueezeDims()
    return op_attr

def parse_split_options(op_opt):
    option = tflite.SplitOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["num_splits"] = option.Numsplits()
    return op_attr

def parse_splitv_options(op_opt):
    option = tflite.SplitVOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["num_splits"] = option.Numsplits()
    return op_attr

def parse_stridedslice_options(op_opt):
    option = tflite.StridedSliceOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["begin_mask"] = option.BeginMask()
    op_attr["end_mask"] = option.EndMask()
    op_attr["ellipsis_mask"] = option.EllipsisMask()
    op_attr["new_axis_mask"] = option.NewAxisMask()
    op_attr["shrink_axis_mask"] = option.ShrinkAxisMask()
    return op_attr

def parse_logsoftmax_options(op_opt):
    return parse_null_options(op_opt)

def parse_cast_options(op_opt):
    option = tflite.CastOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["in_data_type"] = option.InDataType()
    op_attr["out_data_type"] = option.OutDataType()
    return op_attr

def parse_gather_options(op_opt):
    return parse_null_options()

def parse_greaterequal_options(op_opt):
    return parse_null_options(op_opt)

def parse_less_options(op_opt):
    return parse_null_options(op_opt)

def parse_lessequal_options(op_opt):
    return parse_null_options(op_opt)

def parse_neg_options(op_opt):
    return parse_null_options(op_opt)

def parse_select_options(op_opt):
    return parse_null_options(op_opt)

def parse_slice_options(op_opt):
    return parse_null_options(op_opt)

def parse_transposeconv_options(op_opt):
    option = tflite.TransposeConvOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["padding"] = TFLITE_PADDING_TYPE_STR_MAP[option.Padding()]
    op_attr["stride_w"] = option.StrideW()
    op_attr["stride_h"] = option.StrideH()
    return op_attr

def parse_expand_options(op_opt):
    return parse_null_options(op_opt)

def parse_sparsetodense_options(op_opt):
    option = tflite.SparseToDenseOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["validate_indices"] = option.ValidateIndices()
    return op_attr

def parse_equal_options(op_opt):
    return parse_null_options(op_opt)

def parse_notequal_options(op_opt):
    return parse_null_options(op_opt)

def parse_shape_options(op_opt):
    option = tflite.ShapeOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["out_type"] = option.OutType()
    return op_attr

def parse_rank_options(op_opt):
    return parse_null_options(op_opt)

def parse_pow_options(op_opt):
    return parse_null_options(op_opt)

def parse_fakequant_options(op_opt):
    option = tflite.FakeQuantOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["min"] = option.Min()
    op_attr["max"] = option.Max()
    op_attr["num_bits"] = option.NumBits()
    op_attr["narrow_range"] = option.NarrowRange()
    return op_attr

def parse_pack_options(op_opt):
    option = tflite.PackOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["values_count"] = option.ValuesCount()
    op_attr["axis"] = option.Axis()
    return op_attr

def parse_logicalor_options(op_opt):
    return parse_null_options(op_opt)

def parse_oneshot_options(op_opt):
    option = tflite.OneHotOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["axis"] = option.Axis()
    return op_attr

def parse_abs_options(op_opt):
    return parse_null_options(op_opt)

def parse_hardswish_options(op_opt):
    return parse_null_options(op_opt)

def parse_logicaland_options(op_opt):
    return parse_null_options(op_opt)

def parse_logicalnot_options(op_opt):
    return parse_null_options(op_opt)

def parse_unpack_options(op_opt):
    option = tflite.UnpackOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["num"] = option.Num()
    op_attr["axis"] = option.Axis()
    return op_attr

def parse_floordic_options(op_opt):
    return parse_null_options(op_opt)

def parse_square_options(op_opt):
    return parse_null_options(op_opt)

def parse_zeroslike_options(op_opt):
    return parse_null_options(op_opt)

def parse_fill_options(op_opt):
    return parse_null_options(op_opt)

def parse_floormod_options(op_opt):
    return parse_null_options(op_opt)

def parse_range_options(op_opt):
    return parse_null_options(op_opt)

def parse_leakyrelu_options(op_opt):
    option = tflite.LeakyReluOptions()
    option.Init(op_opt.Bytes, op_opt.Pos)
    op_attr = {}
    op_attr["alpha"] = option.Alpha()
    return op_attr


class TfliteModelParser():
    def __init__(self, model, subgraph):
        self.model = model
        self.subgraph = subgraph
        self.TFLITE_OPTION_PARSER = {}
        # builtin ops
        self.TFLITE_OPTION_PARSER["ADD"] = parse_add_options
        self.TFLITE_OPTION_PARSER["AVERAGE_POOL_2D"] = parse_pool2d_options
        self.TFLITE_OPTION_PARSER["CONCATENATION"] = parse_concatenation_options
        self.TFLITE_OPTION_PARSER["CONV_2D"] = parse_conv2d_options
        self.TFLITE_OPTION_PARSER["DEPTHWISE_CONV_2D"] = parse_depthwiseconv2d_options
        self.TFLITE_OPTION_PARSER["DEPTH_TO_SPACE"] = parse_depthtospace_options
        self.TFLITE_OPTION_PARSER["FLOOR"] = parse_null_options
        self.TFLITE_OPTION_PARSER["FULLY_CONNECTED"] = parse_fullyconnected_options
        self.TFLITE_OPTION_PARSER["MAX_POOL_2D"] = parse_pool2d_options
        self.TFLITE_OPTION_PARSER["MUL"] = parse_mul_options
        self.TFLITE_OPTION_PARSER["RELU"] = parse_null_options
        self.TFLITE_OPTION_PARSER["RELU6"] = parse_null_options
        self.TFLITE_OPTION_PARSER["TANH"] = parse_null_options
        self.TFLITE_OPTION_PARSER["SIGMOID"] = parse_null_options
        self.TFLITE_OPTION_PARSER["RELU_N1_TO_1"] = parse_null_options
        self.TFLITE_OPTION_PARSER["RESHAPE"] = parse_reshape_options
        self.TFLITE_OPTION_PARSER["RESIZE_BILINEAR"] = parse_resizebilinear_options
        self.TFLITE_OPTION_PARSER["SOFTMAX"] = parse_softmax_options
        self.TFLITE_OPTION_PARSER["SPACE_TO_DEPTH"] = parse_spacetodepth_options
        self.TFLITE_OPTION_PARSER["TRANSPOSE"] = parse_transpose_options
        self.TFLITE_OPTION_PARSER["MEAN"] = parse_null_options
        self.TFLITE_OPTION_PARSER["SUB"] = parse_sub_options
        self.TFLITE_OPTION_PARSER["DIV"] = parse_div_options
        self.TFLITE_OPTION_PARSER["SQUEEZE"] = parse_squeeze_options
        self.TFLITE_OPTION_PARSER["PAD"] = parse_pad_options
        self.TFLITE_OPTION_PARSER["PADV2"] = parse_padv2_options
        self.TFLITE_OPTION_PARSER["CAST"] = parse_cast_options
        self.TFLITE_OPTION_PARSER["PRELU"] = parse_null_options
        self.TFLITE_OPTION_PARSER["SHAPE"] = parse_shape_options
        self.TFLITE_OPTION_PARSER["POW"] = parse_pow_options
        self.TFLITE_OPTION_PARSER["SPLIT"] = parse_split_options
        self.TFLITE_OPTION_PARSER["SLICE"] = parse_slice_options
        self.TFLITE_OPTION_PARSER["RESIZE_NEAREST_NEIGHBOR"] = parse_resizenearesetneighbor_options
        self.TFLITE_OPTION_PARSER["QUANTIZE"] = parse_null_options
        self.TFLITE_OPTION_PARSER["DEQUANTIZE"] = parse_null_options
        # custom ops
        self.TFLITE_OPTION_PARSER["swish"] = parse_null_options


    def tensor_type_to_string(self, tensor_type):
        if tensor_type == tflite.TensorType.FLOAT16:   return "FLOAT16"
        elif tensor_type == tflite.TensorType.FLOAT32: return "FLOAT32"
        elif tensor_type == tflite.TensorType.FLOAT64: return "FLOAT64"
        elif tensor_type == tflite.TensorType.UINT8:   return "UINT8"
        elif tensor_type == tflite.TensorType.INT8:    return "INT8"
        elif tensor_type == tflite.TensorType.UINT16:  return "UINT16"
        elif tensor_type == tflite.TensorType.INT16:   return "INT16"
        elif tensor_type == tflite.TensorType.UINT32:  return "UINT32"
        elif tensor_type == tflite.TensorType.INT32:   return "INT32"
        elif tensor_type == tflite.TensorType.UINT64:  return "UINT64"
        elif tensor_type == tflite.TensorType.INT64:   return "INT64"
        else: assert False,"convert tensor type {} to string fail".format(str(tensor_type))


    def tensor_type_to_np_type(self, tensor_type):
        if tensor_type == tflite.TensorType.FLOAT16:   return np.float16
        elif tensor_type == tflite.TensorType.FLOAT32: return np.float32
        elif tensor_type == tflite.TensorType.FLOAT64: return np.float64
        elif tensor_type == tflite.TensorType.UINT8:   return np.uint8
        elif tensor_type == tflite.TensorType.INT8:    return np.int8
        elif tensor_type == tflite.TensorType.UINT16:  return np.uint16
        elif tensor_type == tflite.TensorType.INT16:   return np.int16
        elif tensor_type == tflite.TensorType.UINT32:  return np.uint32
        elif tensor_type == tflite.TensorType.INT32:   return np.int32
        elif tensor_type == tflite.TensorType.UINT64:  return np.uint64
        elif tensor_type == tflite.TensorType.INT64:   return np.int64
        else: assert False,"convert tensor type {} to np type fail".format(str(tensor_type))


    def get_op_attrs(self, op_type, tflite_op):
        op_opt = tflite_op.BuiltinOptions()
        assert op_type in self.TFLITE_OPTION_PARSER.keys(), "current not support {} option".format(op_type)
        return self.TFLITE_OPTION_PARSER[op_type](op_opt)


    def construct_tflite_op(self, op_type, op_index, log_flag = False):
        op_info = {}
        # use op index as op name
        op_info["name"] = str(op_index)
        # op_info["location"] = op_index
        op_info["type"] = op_type
        op_info["attr"] = {}
        assert op_index >= 0 and op_index < self.subgraph.OperatorsLength(), "invalid op_index:{} for {}".format(op_index, op_type)
        tflite_op = self.subgraph.Operators(op_index)
        inputs = list(tflite_op.InputsAsNumpy())
        outputs = list(tflite_op.OutputsAsNumpy())
        op_info["inputs"] = inputs
        op_info["outputs"] = outputs
        op_info["attr"] = self.get_op_attrs(op_type, tflite_op)
        return op_info


    def parse_tflite_ops(self):
        tflite_ops = []
        # operators
        for index in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(index)
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            if opcode in tflite.BUILTIN_OPCODE2NAME:
                op_type = tflite.opcode2name(opcode)
            else:
                raise ValueError("Opcode {} is not a TFLite builtin operator!".format(opcode))
            if op_type == "CUSTOM":
                op_type = self.model.OperatorCodes(op.OpcodeIndex()).CustomCode().decode("utf-8")
            tflite_ops.append(self.construct_tflite_op(op_type, index))
        return tflite_ops


    def parse_tflite_tensors(self):
        model_tensors = []
        # tensors
        for index in range(self.subgraph.TensorsLength()):
            tensor_info = {}
            tensor = self.subgraph.Tensors(index)
            # get tensor base info
            tensor_info["name"] = tensor.Name().decode("utf-8")
            tensor_info["index"] = index
            tensor_info["shape"]  = tensor.ShapeAsNumpy().tolist()
            tensor_info["type"]   = self.tensor_type_to_string(tensor.Type())

            # get tensro quantization info
            quantization = tensor.Quantization()
            if quantization != None:
                assert type(quantization) == tflite.QuantizationParameters
                quant_info = {}
                quant_info["scale"] = quantization.ScaleAsNumpy().tolist() if quantization.ScaleLength() > 0 else []
                quant_info["max"] = quantization.MaxAsNumpy().tolist() if quantization.MaxLength() > 0 else []
                quant_info["min"] = quantization.MinAsNumpy().tolist() if quantization.MinLength() > 0 else []
                quant_info["zero_point"] = quantization.ZeroPointAsNumpy().tolist() if quantization.ZeroPointLength() > 0 else []
                quant_info["quantized_dimension"] = quantization.QuantizedDimension()
                assert len(quant_info["scale"]) > 0, "quant's scale size must > 0"
                if len(quant_info["zero_point"]) == 0:
                    quant_info["zero_point"] = [0]*len(quant_info["scale"])
                tensor_info["quantization"] = quant_info
            else:
                tensor_info["quantization"] = None

            # get tensor buffer
            buffer_idx = tensor.Buffer()
            tensor_shape = tensor.ShapeAsNumpy()
            assert(buffer_idx < self.model.BuffersLength())
            raw_data = self.model.Buffers(buffer_idx).DataAsNumpy()
            if isinstance(raw_data, int) and raw_data == 0:
                tensor_data = None
            else:
                np_type = self.tensor_type_to_np_type(tensor.Type())
                tensor_data = np.frombuffer(raw_data, dtype=np_type)
                if len(tensor_shape) > 0:
                    tensor_data = tensor_data.reshape(tensor_shape)
            tensor_info["buffer"] = tensor_data
            model_tensors.append(tensor_info)
        return model_tensors


    def parse_tflite_inputs(self):
        model_inputs = []
        # inputs
        for i in range(self.subgraph.InputsLength()):
            # FIXME: assert they have been created.
            index = self.subgraph.Inputs(i)
            model_inputs.append(index)
        return model_inputs


    def parse_tflite_outputs(self):
        model_outputs = []
        # outputs
        for i in range(self.subgraph.OutputsLength()):
            index = self.subgraph.Outputs(i)
            model_outputs.append(index)
        return model_outputs


def parse_tflite_model(tflite_model_data):
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_data, 0)
    graph_count = tflite_model.SubgraphsLength()
    if (graph_count != 1):
        raise NotImplementedError("expect 1 sub_graph but get {}", graph_count)
    tflite_subgraph = tflite_model.Subgraphs(0)
    model_info = {}
    model_parser = TfliteModelParser(tflite_model, tflite_subgraph)
    model_info["inputs"] = model_parser.parse_tflite_inputs()
    model_info["outputs"] = model_parser.parse_tflite_outputs()
    model_info["tensors"] = model_parser.parse_tflite_tensors()
    model_info["nodes"] = model_parser.parse_tflite_ops()
    model_info["insert_nodes"] = []
    return model_info