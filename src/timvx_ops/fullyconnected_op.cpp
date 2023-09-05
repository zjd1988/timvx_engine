/***********************************
******  fullconnected_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/fullyconnected.h"
#include "timvx_ops/fullyconnected_op.h"

namespace TimVX
{

    bool FullyConnectedOpCreator::parseAxisAttr(const json& op_info, FullyConnectedOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool FullyConnectedOpCreator::parseWeightsAttr(const json& op_info, FullyConnectedOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "weights", op_attr.weights, false);
    }

    bool FullyConnectedOpCreator::parseOpAttr(const json& op_info, FullyConnectedOpAttr& op_attr)
    {
        op_attr.weights = 0;
        return parseAxisAttr(op_info, op_attr) && parseWeightsAttr(op_info, op_attr);
    }

    Operation* FullyConnectedOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        FullyConnectedOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t axis    = op_attr.axis;
        uint32_t weights = op_attr.weights;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, weights);
        return graph->CreateOperation<ops::FullyConnected>(axis, weights).get();
    }

    REGISTER_OP_CREATOR(FullyConnectedOpCreator, FullyConnected);

} // namespace TimVX