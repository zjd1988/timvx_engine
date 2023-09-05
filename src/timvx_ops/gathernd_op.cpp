/***********************************
******  gathernd_op.cpp
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#include "tim/vx/ops/gathernd.h"
#include "timvx_ops/gathernd_op.h"

namespace TimVX
{
    bool GatherNdOpCreator::parseOpAttr(const json& op_info, GatherNdOpAttr& op_attr)
    {
        return true;
    }

    Operation* GatherNdOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        GatherNdOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        return graph->CreateOperation<ops::GatherNd>().get();
    }

    REGISTER_OP_CREATOR(GatherNdOpCreator, GatherNd);

} // namespace TimVX