/***********************************
******  select_op.cpp
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#include "tim/vx/ops/select.h"
#include "timvx_ops/select_op.h"

namespace TimVX
{
    bool SelectOpCreator::parseOpAttr(const json& op_info, SelectOpAttr& op_attr)
    {
        return true;
    }

    Operation* SelectOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        SelectOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        return graph->CreateOperation<ops::Select>().get();
    }

    REGISTER_OP_CREATOR(SelectOpCreator, Select);

} // namespace TimVX