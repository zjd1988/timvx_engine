/***********************************
******  softmax_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/softmax.h"
#include "timvx_ops/softmax_op.h"

namespace TimVX
{

    bool SoftmaxOpCreator::parseBetaAttr(const json& op_info, SoftmaxOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "beta", op_attr.beta);
    }

    bool SoftmaxOpCreator::parseAxisAttr(const json& op_info, SoftmaxOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool SoftmaxOpCreator::parseOpAttr(const json& op_info, SoftmaxOpAttr& op_attr)
    {
        return parseBetaAttr(op_info, op_attr) && parseAxisAttr(op_info, op_attr);
    }

    Operation* SoftmaxOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        SoftmaxOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        float beta   = op_attr.beta;
        int32_t axis = op_attr.axis;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, beta);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        return graph->CreateOperation<ops::Softmax>(beta, axis).get();
    }

    REGISTER_OP_CREATOR(SoftmaxOpCreator, Softmax);

} // namespace TimVX