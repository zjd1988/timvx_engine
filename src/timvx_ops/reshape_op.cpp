/***********************************
******  reshape_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/reshape.h"
#include "timvx_ops/reshape_op.h"

namespace TimVX
{

    bool ReshapeOpCreator::parseSizeAttr(const json& op_info, ReshapeOpAttr& op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "size", op_attr.size);
    }

    bool ReshapeOpCreator::parseOpAttr(const json& op_info, ReshapeOpAttr& op_attr)
    {
        return parseSizeAttr(op_info, op_attr);
    }

    Operation* ReshapeOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ReshapeOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<uint32_t> size = op_attr.size;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, size);
        return graph->CreateOperation<ops::Reshape>(size).get();
    }

    REGISTER_OP_CREATOR(ReshapeOpCreator, Reshape);

} // namespace TimVX