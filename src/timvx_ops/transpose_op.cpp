/***********************************
******  transpose_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/transpose.h"
#include "timvx_ops/transpose_op.h"

namespace TimVX
{

    bool TransposeOpCreator::parsePermAttr(const json& op_info, TransposeOpAttr& op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "perm", op_attr.perm);
    }

    bool TransposeOpCreator::parseOpAttr(const json& op_info, TransposeOpAttr& op_attr)
    {
        return parsePermAttr(op_info, op_attr);
    }

    Operation* TransposeOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        TransposeOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<uint32_t> perm = op_attr.perm;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, perm);
        return graph->CreateOperation<ops::Transpose>(perm).get();
    }

    REGISTER_OP_CREATOR(TransposeOpCreator, Transpose);

} // namespace TimVX