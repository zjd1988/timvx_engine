/***********************************
******  dropout_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/dropout.h"
#include "timvx_ops/dropout_op.h"

namespace TimVX
{

    bool DropoutOpCreator::parseRatioAttr(const json& op_info, DropoutOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "ratio", op_attr.ratio);
    }

    bool DropoutOpCreator::parseOpAttr(const json& op_info, DropoutOpAttr& op_attr)
    {
        return parseRatioAttr(op_info, op_attr);
    }

    Operation* DropoutOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        DropoutOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        float ratio = op_attr.ratio;
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, ratio);
        return graph->CreateOperation<ops::Dropout>(ratio).get();
    }

    REGISTER_OP_CREATOR(DropoutOpCreator, Dropout);

} // namespace TimVX