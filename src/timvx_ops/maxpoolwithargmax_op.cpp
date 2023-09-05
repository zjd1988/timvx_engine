/***********************************
******  maxpoolwithargmax_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/maxpoolwithargmax.h"
#include "timvx_ops/maxpoolwithargmax_op.h"

namespace TimVX
{

    bool MaxpoolWithArgmaxOpCreator::parsePaddingAttr(const json& op_info, MaxpoolWithArgmaxOpAttr& op_attr)
    {
        return parsePadType(op_info, m_op_name, "padding", op_attr.padding);
    }

    bool MaxpoolWithArgmaxOpCreator::parseKsizeAttr(const json& op_info, MaxpoolWithArgmaxOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "ksize", op_attr.ksize);
    }

    bool MaxpoolWithArgmaxOpCreator::parseStrideAttr(const json& op_info, MaxpoolWithArgmaxOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool MaxpoolWithArgmaxOpCreator::parseRoundTypeAttr(const json& op_info, MaxpoolWithArgmaxOpAttr& op_attr)
    {
        return parseRoundType(op_info, m_op_name, "round_type", op_attr.round_type, false);
    }

    bool MaxpoolWithArgmaxOpCreator::parseLayoutAttr(const json& op_info, MaxpoolWithArgmaxOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout, false);
    }

    bool MaxpoolWithArgmaxOpCreator::parseOpAttr(const json& op_info, MaxpoolWithArgmaxOpAttr& op_attr)
    {
        op_attr.round_type = RoundType::FLOOR;
        op_attr.layout = DataLayout::WHCN; // always set WHCN
        return parsePaddingAttr(op_info, op_attr) && parseKsizeAttr(op_info, op_attr) && 
            parseStrideAttr(op_info, op_attr) && parseRoundTypeAttr(op_info, op_attr) && 
            parseLayoutAttr(op_info, op_attr);
    }

    Operation* MaxpoolWithArgmaxOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        MaxpoolWithArgmaxOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        PadType                 padding        = op_attr.padding;
        std::array<uint32_t, 2> ksize          = op_attr.ksize;
        std::array<uint32_t, 2> stride         = op_attr.stride;
        RoundType               round_type     = op_attr.round_type;
        DataLayout              layout         = op_attr.layout;

        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, padding, gPadTypeToStrMap[padding]);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, ksize);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, stride);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, round_type, gRoundTypeToStrMap[round_type]);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, layout, gDataLayoutToStrMap[layout]);
        return graph->CreateOperation<ops::MaxpoolWithArgmax>(padding, ksize, stride, 
            round_type, layout).get();
    }

    REGISTER_OP_CREATOR(MaxpoolWithArgmaxOpCreator, MaxpoolWithArgmax);

} // namespace TimVX