/***********************************
******  matmul_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/matmul.h"
#include "timvx_ops/matmul_op.h"

namespace TimVX
{

    bool MatmulOpCreator::parseTransposeAAttr(const json& op_info, MatmulOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "transpose_a", op_attr.transpose_a, false);
    }

    bool MatmulOpCreator::parseTransposeBAttr(const json& op_info, MatmulOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "transpose_b", op_attr.transpose_b, false);
    }

    bool MatmulOpCreator::parseAdjointAAttr(const json& op_info, MatmulOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "adjoint_a", op_attr.adjoint_a, false);
    }

    bool MatmulOpCreator::parseAdjointBAttr(const json& op_info, MatmulOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "adjoint_b", op_attr.adjoint_b, false);
    }

    bool MatmulOpCreator::parseOpAttr(const json& op_info, MatmulOpAttr& op_attr)
    {
        op_attr.transpose_a = false;
        op_attr.transpose_b = false;
        op_attr.adjoint_a = false;
        op_attr.adjoint_b = false;
        return parseTransposeAAttr(op_info, op_attr) && parseTransposeBAttr(op_info, op_attr) && 
            parseAdjointAAttr(op_info, op_attr) && parseAdjointBAttr(op_info, op_attr);
    }

    Operation* MatmulOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        MatmulOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        bool transpose_a = op_attr.transpose_a;
        bool transpose_b = op_attr.transpose_b;
        bool adjoint_a = op_attr.adjoint_a;
        bool adjoint_b = op_attr.adjoint_b;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, transpose_a);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, transpose_b);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, adjoint_a);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, adjoint_b);
        return graph->CreateOperation<ops::Matmul>(transpose_a, transpose_b, adjoint_a, adjoint_b).get();
    }

    REGISTER_OP_CREATOR(MatmulOpCreator, Matmul);

} // namespace TimVX