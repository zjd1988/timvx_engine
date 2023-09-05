/***********************************
******  l2normalization_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/l2normalization.h"
#include "timvx_ops/l2normalization_op.h"

namespace TimVX
{

    bool L2NormalizationOpCreator::parseAxisAttr(const json& op_info, L2NormalizationOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool L2NormalizationOpCreator::parseOpAttr(const json& op_info, L2NormalizationOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr);
    }

    Operation* L2NormalizationOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        L2NormalizationOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        int32_t axis = op_attr.axis;
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        return graph->CreateOperation<ops::L2Normalization>(axis).get();
    }

    REGISTER_OP_CREATOR(L2NormalizationOpCreator, L2Normalization);

} // namespace TimVX