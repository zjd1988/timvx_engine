/***********************************
******  reorg_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/reorg.h"
#include "timvx_ops/reorg_op.h"

namespace TimVX
{

    bool ReorgOpCreator::parseStrideAttr(const json& op_info, ReorgOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool ReorgOpCreator::parseOpAttr(const json& op_info, ReorgOpAttr& op_attr)
    {
        return parseStrideAttr(op_info, op_attr);
    }

    Operation* ReorgOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ReorgOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t stride = op_attr.stride;
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, stride);
        return graph->CreateOperation<ops::Reorg>(stride).get();
    }

    REGISTER_OP_CREATOR(ReorgOpCreator, Reorg);

} // namespace TimVX