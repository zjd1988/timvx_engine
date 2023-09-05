/***********************************
******  logsoftmax_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/logsoftmax.h"
#include "timvx_ops/logsoftmax_op.h"

namespace TimVX
{

    bool LogSoftmaxOpCreator::parseAxisAttr(const json& op_info, LogSoftmaxOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool LogSoftmaxOpCreator::parseBetaAttr(const json& op_info, LogSoftmaxOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "beta", op_attr.beta, false);
    }

    bool LogSoftmaxOpCreator::parseOpAttr(const json& op_info, LogSoftmaxOpAttr& op_attr)
    {
        op_attr.beta = 1.f;
        return parseAxisAttr(op_info, op_attr) && parseBetaAttr(op_info, op_attr);
    }

    Operation* LogSoftmaxOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        LogSoftmaxOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        int32_t axis = op_attr.axis;
        float beta = op_attr.beta;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, beta);
        return graph->CreateOperation<ops::LogSoftmax>(axis, beta).get();
    }

    REGISTER_OP_CREATOR(LogSoftmaxOpCreator, LogSoftmax);

} // namespace TimVX