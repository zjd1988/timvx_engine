/***********************************
******  conv2d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/conv2d.h"
#include "timvx_ops/conv2d_op.h"

namespace TimVX
{

    bool Conv2dOpCreator::parseWeightsAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "weights", op_attr.weights, false);
    }

    bool Conv2dOpCreator::parsePaddingAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parsePadType(op_info, m_op_name, "padding", op_attr.padding, false);
    }

    bool Conv2dOpCreator::parseKsizeAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "ksize", op_attr.ksize, false);
    }

    bool Conv2dOpCreator::parseStrideAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool Conv2dOpCreator::parseDilationAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "dilation", op_attr.dilation);
    }

    bool Conv2dOpCreator::parsePadAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 4>(op_info, m_op_name, "pad", op_attr.pad, false);
    }

    bool Conv2dOpCreator::parseMultiplierAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "multiplier", op_attr.multiplier, false);
    }

    bool Conv2dOpCreator::parseInputLayoutAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "input_layout", op_attr.input_layout, false);
    }
    
    bool Conv2dOpCreator::parseKernelLayoutAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "kernel_layout", op_attr.kernel_layout, false);
    }

    bool Conv2dOpCreator::parseOpAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        op_attr.weights = 0;
        op_attr.padding = PadType::AUTO;
        op_attr.ksize = {0, 0};
        op_attr.multiplier = 0;
        op_attr.pad = {0, 0, 0, 0};
        op_attr.input_layout = DataLayout::WHCN;
        op_attr.kernel_layout = DataLayout::WHIcOc;
        return parseWeightsAttr(op_info, op_attr) && parsePaddingAttr(op_info, op_attr) && 
            parseKsizeAttr(op_info, op_attr) && parseStrideAttr(op_info, op_attr) && 
            parseDilationAttr(op_info, op_attr) && parsePadAttr(op_info, op_attr) && 
            parseMultiplierAttr(op_info, op_attr) && parseInputLayoutAttr(op_info, op_attr) && 
            parseKernelLayoutAttr(op_info, op_attr);
    }

    Operation* Conv2dOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
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

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, weights);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, padding, gPadTypeToStrMap[padding]);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, ksize);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, stride);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, dilation);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, pad);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, multiplier);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, input_layout, gDataLayoutToStrMap[input_layout]);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, kernel_layout, gDataLayoutToStrMap[kernel_layout]);
        return graph->CreateOperation<ops::Conv2d>(weights, padding, ksize, stride, 
            dilation, pad, multiplier, input_layout, kernel_layout).get();
    }

    REGISTER_OP_CREATOR(Conv2dOpCreator, Conv2d);

} // namespace TimVX