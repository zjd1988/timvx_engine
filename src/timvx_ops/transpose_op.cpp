/***********************************
******  transpose_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/transpose.h"
#include "transpose_op.h"


namespace TIMVX
{
    bool TransposeCreator::parseOpAttr(const json &op_info, TransposeOpAttr &op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "perm", op_attr.perm);
    }

    Operation* TransposeCreator::onCreate(std::shared_ptr<Graph> &graph, const json &op_info)
    {
        TransposeOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<uint32_t> perm = op_attr.perm;
        return graph->CreateOperation<ops::Transpose>(perm).get();
    }

    REGISTER_OP_CREATOR(TransposeCreator, Transpose);
} // namespace TIMVX