/***********************************
******  relational_operations_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/relational_operations.h"
#include "timvx_ops/relational_operations_op.h"

namespace TimVX
{

    bool RelationalOperationsOpCreator::parseGreaterAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsOpCreator::parseGreaterOrEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsOpCreator::parseLessAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsOpCreator::parseLessOrEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsOpCreator::parseNotEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsOpCreator::parseEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsOpCreator::parseOpAttr(std::string relational_type, const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        if ("Greater" == relational_type)
            return parseGreaterAttr(op_info, op_attr);
        else if ("GreaterOrEqual" == relational_type)
            return parseGreaterOrEqualAttr(op_info, op_attr);
        else if ("Less" == relational_type)
            return parseLessAttr(op_info, op_attr);
        else if ("LessOrEqual" == relational_type)
            return parseLessOrEqualAttr(op_info, op_attr);
        else if ("NotEqual" == relational_type)
            return parseNotEqualAttr(op_info, op_attr);
        else if ("Equal" == relational_type)
            return parseEqualAttr(op_info, op_attr);
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported relational operations op type: {}", relational_type);
        return false;
    }

    Operation* RelationalOperationsOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        RelationalOperationsOpAttr op_attr;
        std::string relational_type;
        if (!parseValue<std::string>(op_info, m_op_name, "relational_type", relational_type))
            return nullptr;
        if (!parseOpAttr(relational_type, op_info, op_attr))
            return nullptr;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, relational_type);
        if ("Greater" == relational_type)
        {
            return graph->CreateOperation<ops::Greater>().get();
        }
        else if ("GreaterOrEqual" == relational_type)
        {
            return graph->CreateOperation<ops::GreaterOrEqual>().get();
        }
        else if ("Less" == relational_type)
        {
            return graph->CreateOperation<ops::Less>().get();
        }
        else if ("LessOrEqual" == relational_type)
        {
            return graph->CreateOperation<ops::LessOrEqual>().get();
        }
        else if ("NotEqual" == relational_type)
        {
            return graph->CreateOperation<ops::NotEqual>().get();
        }
        else if ("Equal" == relational_type)
        {
            return graph->CreateOperation<ops::Equal>().get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported relational op type: {}", relational_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(RelationalOperationsOpCreator, RelationalOperations);

} // namespace TimVX