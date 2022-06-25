/***********************************
******  concat_op.cpp
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#include "tim/vx/ops/concat.h"
#include "timvx_ops/concat_op.h"

namespace TIMVXPY
{
    bool ConcatCreator::parseOpAttr(const json &op_info, ConcatOpAttr &op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "axis", op_attr.axis) &&
            parseValue<int>(op_info, m_op_name, "input_cnt", op_attr.input_cnt);
    }

    Operation* ConcatCreator::on_create(std::shared_ptr<Graph> &graph, const json &op_info)
    {
        ConcatOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t axis = op_attr.axis;
        int input_cnt = op_attr.input_cnt;
        return graph->CreateOperation<ops::Concat>(axis, input_cnt).get();
    }

    REGISTER_OP_CREATOR(ConcatCreator, Concat);
} // namespace TIMVXPY