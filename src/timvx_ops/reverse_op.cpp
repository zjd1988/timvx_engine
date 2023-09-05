/***********************************
******  reverse_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/reverse.h"
#include "timvx_ops/reverse_op.h"

namespace TimVX
{

    bool ReverseOpCreator::parseAxisAttr(const json& op_info, ReverseOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool ReverseOpCreator::parseOpAttr(const json& op_info, ReverseOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr);
    }

    Operation* ReverseOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ReverseOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> axis = op_attr.axis;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        return graph->CreateOperation<ops::Reverse>(axis).get();
    }

    REGISTER_OP_CREATOR(ReverseOpCreator, Reverse);

} // namespace TimVX