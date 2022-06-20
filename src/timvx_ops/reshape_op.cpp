/***********************************
******  reshape_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/reshape.h"
#include "reshape_op.h"


namespace TIMVXPY
{
    bool ReshapeCreator::parseOpAttr(const json &op_info, ReshapeOpAttr &op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "size", op_attr.size);
    }

    Operation* ReshapeCreator::onCreate(std::shared_ptr<Graph> &graph, const json &op_info)
    {
        ReshapeOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<uint32_t> size = op_attr.size;
        return graph->CreateOperation<ops::Reshape>(size).get();
    }

    REGISTER_OP_CREATOR(ReshapeCreator, Reshape);
} // namespace TIMVXPY