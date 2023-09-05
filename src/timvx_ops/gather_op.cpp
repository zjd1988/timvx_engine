/***********************************
******  gather_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/gather.h"
#include "timvx_ops/gather_op.h"

namespace TimVX
{

    bool GatherOpCreator::parseAxisAttr(const json& op_info, GatherOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool GatherOpCreator::parseOpAttr(const json& op_info, GatherOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr);
    }

    Operation* GatherOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        GatherOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        int32_t axis = op_attr.axis;
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        return graph->CreateOperation<ops::Gather>(axis).get();
    }

    REGISTER_OP_CREATOR(GatherOpCreator, Gather);

} // namespace TimVX