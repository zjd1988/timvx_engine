/***********************************
******  scatternd_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/scatternd.h"
#include "timvx_ops/scatternd_op.h"

namespace TimVX
{

    bool ScatterNDOpCreator::parseShapeAttr(const json& op_info, ScatterNDOpAttr& op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "shape", op_attr.shape);
    }

    bool ScatterNDOpCreator::parseOpAttr(const json& op_info, ScatterNDOpAttr& op_attr)
    {
        return parseShapeAttr(op_info, op_attr);
    }

    Operation* ScatterNDOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ScatterNDOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<uint32_t> shape = op_attr.shape;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, shape);
        return graph->CreateOperation<ops::ScatterND>(shape).get();
    }

    REGISTER_OP_CREATOR(ScatterNDOpCreator, ScatterND);

} // namespace TimVX