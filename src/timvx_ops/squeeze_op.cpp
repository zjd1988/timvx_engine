/***********************************
******  squeeze_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/squeeze.h"
#include "timvx_ops/squeeze_op.h"

namespace TimVX
{

    bool SqueezeOpCreator::parseAxisAttr(const json& op_info, SqueezeOpAttr& op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool SqueezeOpCreator::parseOpAttr(const json& op_info, SqueezeOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr);
    }

    Operation* SqueezeOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        SqueezeOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<uint32_t> axis = op_attr.axis;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        return graph->CreateOperation<ops::Squeeze>(axis).get();
    }

    REGISTER_OP_CREATOR(SqueezeOpCreator, Squeeze);

} // namespace TimVX