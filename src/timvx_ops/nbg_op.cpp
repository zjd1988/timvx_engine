/***********************************
******  nbg_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/nbg.h"
#include "timvx_ops/nbg_op.h"

namespace TimVX
{

    bool NBGOpCreator::parseBinaryAttr(const json& op_info, NBGOpAttr& op_attr)
    {
        // use size_t to store void* binary
        size_t binary_ptr;
        if (parseValue<size_t>(op_info, m_op_name, "binary", binary_ptr))
            op_attr.binary = (void*)binary_ptr;
        else
            return false;
        return true;
    }

    bool NBGOpCreator::parseInputCountAttr(const json& op_info, NBGOpAttr& op_attr)
    {
        return parseValue<size_t>(op_info, m_op_name, "input_count", op_attr.input_count);
    }

    bool NBGOpCreator::parseOutputCountAttr(const json& op_info, NBGOpAttr& op_attr)
    {
        return parseValue<size_t>(op_info, m_op_name, "output_count", op_attr.output_count);
    }

    bool NBGOpCreator::parseOpAttr(const json& op_info, NBGOpAttr& op_attr)
    {
        op_attr.binary = nullptr;
        op_attr.input_count = 0;
        op_attr.output_count = 0;
        return parseBinaryAttr(op_info, op_attr) && parseInputCountAttr(op_info, op_attr) && 
            parseOutputCountAttr(op_info, op_attr);
    }

    Operation* NBGOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        NBGOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        const char* binary = (const char*)op_attr.binary;
        size_t input_count = op_attr.input_count;
        size_t output_count = op_attr.output_count;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, binary);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, input_count);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, output_count);
        return graph->CreateOperation<ops::NBG>(binary, input_count, output_count).get();
    }

    REGISTER_OP_CREATOR(NBGOpCreator, NBG);

} // namespace TimVX