/***********************************
******  softmax_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/softmax.h"
#include "softmax_op.h"

namespace TIMVX
{
    bool SoftmaxCreator::parseOpAttr(const json &op_info, SoftmaxOpAttr &op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "beta", op_attr.beta) &&
            parseValue<int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    Operation* SoftmaxCreator::onCreate(std::shared_ptr<Graph> &graph, const json &op_info)
    {
        SoftmaxOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        float beta   = op_attr.beta;
        int32_t axis = op_attr.axis;
        return graph->CreateOperation<ops::Softmax>(beta, axis).get();
    }

    REGISTER_OP_CREATOR(SoftmaxCreator, Softmax);
} // namespace TIMVX