/***********************************
******  instancenormalization_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/instancenormalization.h"
#include "timvx_ops/instancenormalization_op.h"

namespace TimVX
{

    bool InstanceNormalizationOpCreator::parseEpsAttr(const json& op_info, InstanceNormalizationOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "eps", op_attr.eps, false);
    }

    bool InstanceNormalizationOpCreator::parseOpAttr(const json& op_info, InstanceNormalizationOpAttr& op_attr)
    {
        op_attr.eps = 1e-5f;
        return parseEpsAttr(op_info, op_attr);
    }

    Operation* InstanceNormalizationOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        InstanceNormalizationOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        float eps = op_attr.eps;
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, eps);
        return graph->CreateOperation<ops::InstanceNormalization>(eps).get();
    }

    REGISTER_OP_CREATOR(InstanceNormalizationOpCreator, InstanceNormalization);

} // namespace TimVX