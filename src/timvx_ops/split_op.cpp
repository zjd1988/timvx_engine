/***********************************
******  split_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/split.h"
#include "timvx_ops/split_op.h"

namespace TimVX
{

    bool SplitOpCreator::parseAxisAttr(const json& op_info, SplitOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool SplitOpCreator::parseSlicesAttr(const json& op_info, SplitOpAttr& op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "slices", op_attr.slices);
    }

    bool SplitOpCreator::parseOpAttr(const json& op_info, SplitOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr) && parseSlicesAttr(op_info, op_attr);
    }

    Operation* SplitOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        SplitOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t              axis   = op_attr.axis;
        std::vector<uint32_t> slices = op_attr.slices;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, slices);
        return graph->CreateOperation<ops::Split>(axis, slices).get();
    }

    REGISTER_OP_CREATOR(SplitOpCreator, Split);

} // namespace TimVX