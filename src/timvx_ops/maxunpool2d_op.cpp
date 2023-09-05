/***********************************
******  maxunpool2d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/maxunpool2d.h"
#include "timvx_ops/maxunpool2d_op.h"

namespace TimVX
{

    bool MaxUnpool2dOpCreator::parseKsizeAttr(const json& op_info, MaxUnpool2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "ksize", op_attr.ksize);
    }

    bool MaxUnpool2dOpCreator::parseStrideAttr(const json& op_info, MaxUnpool2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool MaxUnpool2dOpCreator::parseLayoutAttr(const json& op_info, MaxUnpool2dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout, false);
    }

    bool MaxUnpool2dOpCreator::parseOpAttr(const json& op_info, MaxUnpool2dOpAttr& op_attr)
    {
        op_attr.layout = DataLayout::WHCN; // always set WHCN
        return parseKsizeAttr(op_info, op_attr) && parseStrideAttr(op_info, op_attr) && 
            parseLayoutAttr(op_info, op_attr);
    }

    Operation* MaxUnpool2dOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        MaxUnpool2dOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::array<uint32_t, 2> ksize          = op_attr.ksize;
        std::array<uint32_t, 2> stride         = op_attr.stride;
        DataLayout              layout         = op_attr.layout;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, ksize);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, stride);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, layout, gDataLayoutToStrMap[layout]);
        return graph->CreateOperation<ops::MaxUnpool2d>(ksize, stride, layout).get();
    }

    REGISTER_OP_CREATOR(MaxUnpool2dOpCreator, MaxUnpool2d);

} // namespace TimVX