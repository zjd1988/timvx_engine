/***********************************
******  stack_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/stack.h"
#include "timvx_ops/stack_op.h"

namespace TimVX
{

    bool StackOpCreator::parseAxisAttr(const json& op_info, StackOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool StackOpCreator::parseInputCntAttr(const json& op_info, StackOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "input_cnt", op_attr.input_cnt);
    }

    bool StackOpCreator::parseOpAttr(const json& op_info, StackOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr) && parseInputCntAttr(op_info, op_attr);
    }

    Operation* StackOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        StackOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t axis         = op_attr.axis;
        int32_t input_cnt     = op_attr.input_cnt;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, input_cnt);
        return graph->CreateOperation<ops::Stack>(axis, input_cnt).get();
    }

    REGISTER_OP_CREATOR(StackOpCreator, Stack);

} // namespace TimVX