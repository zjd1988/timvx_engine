/***********************************
******  unstack_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/unstack.h"
#include "timvx_ops/unstack_op.h"

namespace TimVX
{

    bool UnstackOpCreator::parseAxisAttr(const json& op_info, UnstackOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool UnstackOpCreator::parseOutputNumAttr(const json& op_info, UnstackOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "output_num", op_attr.output_num);
    }

    bool UnstackOpCreator::parseOpAttr(const json& op_info, UnstackOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr) && parseOutputNumAttr(op_info, op_attr);
    }

    Operation* UnstackOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        UnstackOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        int32_t axis         = op_attr.axis;
        uint32_t output_num  = op_attr.output_num;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, output_num);
        return graph->CreateOperation<ops::Unstack>(axis, output_num).get();
    }

    REGISTER_OP_CREATOR(UnstackOpCreator, Unstack);

} // namespace TimVX