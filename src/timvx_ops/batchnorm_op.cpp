/***********************************
******  batchnorm_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/batchnorm.h"
#include "timvx_ops/batchnorm_op.h"

namespace TimVX
{

    bool BatchNormOpCreator::parseEpsAttr(const json& op_info, BatchNormOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "eps", op_attr.eps);
    }

    bool BatchNormOpCreator::parseOpAttr(const json& op_info, BatchNormOpAttr& op_attr)
    {
        return parseEpsAttr(op_info, op_attr);
    }

    Operation* BatchNormOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        BatchNormOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        float eps = op_attr.eps;
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, eps);
        return graph->CreateOperation<ops::BatchNorm>(eps).get();
    }

    REGISTER_OP_CREATOR(BatchNormOpCreator, BatchNorm);

} // namespace TimVX