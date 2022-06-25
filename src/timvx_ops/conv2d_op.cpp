/***********************************
******  conv2d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/conv2d.h"
#include "timvx_ops/conv2d_op.h"

namespace TIMVXPY
{

    bool Conv2dCreator::parseWeights(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "weights", op_attr.weights, false);
    }

    bool Conv2dCreator::parsePadding(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parsePadType(op_info, m_op_name, "padding", op_attr.padding, false);
    }

    bool Conv2dCreator::parseKsize(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "ksize", op_attr.ksize, false);
    }

    bool Conv2dCreator::parse_stride(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool Conv2dCreator::parseDilation(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "dilation", op_attr.dilation);
    }

    bool Conv2dCreator::parsePad(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parseFixList<uint32_t, 4>(op_info, m_op_name, "pad", op_attr.pad, false);
    }

    bool Conv2dCreator::parse_multiplier(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "multiplier", op_attr.multiplier, false);
    }

    bool Conv2dCreator::parseInputLayout(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "input_layout", op_attr.input_layout, false);
    }
    
    bool Conv2dCreator::parseKernelLayout(const json &op_info, Conv2dOpAttr &op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "kernel_layout", op_attr.input_layout, false);
    }

    bool Conv2dCreator::parseOpAttr(const json &op_info, Conv2dOpAttr &op_attr)
    {
        op_attr.weights = 0;
        op_attr.padding = PadType::AUTO;
        op_attr.ksize = {0, 0};
        op_attr.multiplier = 0;
        op_attr.pad = {0, 0, 0, 0};
        op_attr.input_layout = DataLayout::WHCN;
        op_attr.kernel_layout = DataLayout::WHIcOc;
        return parseWeights(op_info, op_attr) && parsePadding(op_info, op_attr)
            && parseKsize(op_info, op_attr) && parseStride(op_info, op_attr)
            && parseDilation(op_info, op_attr) && parsePad(op_info, op_attr)
            && parseMultiplier(op_info, op_attr) && parseInputLayout(op_info, op_attr)
            && parseKernelLayout(op_info, op_attr);
    }

    Operation* Conv2dCreator::onCreate(std::shared_ptr<Graph> &graph, const json &op_info)
    {
        Conv2dOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t                weights        = op_attr.weights;
        PadType                 padding        = op_attr.padding;
        std::array<uint32_t, 2> ksize          = op_attr.ksize;
        std::array<uint32_t, 2> stride         = op_attr.stride;
        std::array<uint32_t, 2> dilation       = op_attr.dilation;
        std::array<uint32_t, 4> pad            = op_attr.pad;
        int32_t                 multiplier     = op_attr.multiplier;
        DataLayout              input_layout   = op_attr.input_layout;
        DataLayout              kernel_layout  = op_attr.kernel_layout;
        TIMVX_PRINT("conv2d op weights: %d\n", weights);
        TIMVX_PRINT("conv2d op padding: %d\n", (int)padding);
        TIMVX_PRINT("conv2d op ksize: [%d, %d]\n", ksize[0], ksize[1]);
        TIMVX_PRINT("conv2d op stride: [%d, %d]\n", stride[0], stride[1]);
        TIMVX_PRINT("conv2d op dilation: [%d, %d]\n", dilation[0], dilation[1]);
        TIMVX_PRINT("conv2d op pad: [%d, %d, %d, %d]\n", pad[0], pad[1], pad[2], pad[3]);
        TIMVX_PRINT("conv2d op multiplier: %d\n", multiplier);
        TIMVX_PRINT("conv2d op input_layout: %d\n", (int)input_layout);
        TIMVX_PRINT("conv2d op kernel_layout: %d\n", (int)kernel_layout);
        return graph->CreateOperation<ops::Conv2d>(weights, padding, ksize, stride, 
            dilation, pad, multiplier, input_layout, kernel_layout).get();
    }

    REGISTER_OP_CREATOR(Conv2dCreator, Conv2d);
} // namespace TIMVXPY