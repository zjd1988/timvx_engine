/***********************************
******  concat_op.cpp
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#include "tim/vx/ops/concat.h"
#include "timvx_ops/concat_op.h"

namespace TimVX
{

    bool ConcatOpCreator::parseAxisAttr(const json& op_info, ConcatOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool ConcatOpCreator::parseInputCntAttr(const json& op_info, ConcatOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "input_cnt", op_attr.input_cnt);
    }

    bool ConcatOpCreator::parseOpAttr(const json& op_info, ConcatOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr) && parseInputCntAttr(op_info, op_attr);
    }

    Operation* ConcatOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ConcatOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t axis = op_attr.axis;
        int input_cnt = op_attr.input_cnt;
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, input_cnt);
        return graph->CreateOperation<ops::Concat>(axis, input_cnt).get();
    }

    REGISTER_OP_CREATOR(ConcatOpCreator, Concat);

} // namespace TimVX