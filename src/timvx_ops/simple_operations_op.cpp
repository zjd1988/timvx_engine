/***********************************
******  simple_operations_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/simple_operations.h"
#include "timvx_ops/simple_operations_op.h"

namespace TimVX
{

    bool SimpleOperationsOpCreator::parseDataConvertAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseNegAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseAbsAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseSinAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseExpAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseLogAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseSqrtAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseRsqrtAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseSquareAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseLogicalNotAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseFloorAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseCastAttr(const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool SimpleOperationsOpCreator::parseOpAttr(std::string simple_type, const json& op_info, SimpleOperationsOpAttr& op_attr)
    {
        if ("DataConvert" == simple_type)
            return parseDataConvertAttr(op_info, op_attr);
        else if ("Neg" == simple_type)
            return parseNegAttr(op_info, op_attr);
        else if ("Abs" == simple_type)
            return parseAbsAttr(op_info, op_attr);
        else if ("Sin" == simple_type)
            return parseSinAttr(op_info, op_attr);
        else if ("Exp" == simple_type)
            return parseExpAttr(op_info, op_attr);
        else if ("Log" == simple_type)
            return parseLogAttr(op_info, op_attr);
        else if ("Sqrt" == simple_type)
            return parseSqrtAttr(op_info, op_attr);
        else if ("Rsqrt" == simple_type)
            return parseRsqrtAttr(op_info, op_attr);
        else if ("Square" == simple_type)
            return parseSquareAttr(op_info, op_attr);
        else if ("LogicalNot" == simple_type)
            return parseLogicalNotAttr(op_info, op_attr);
        else if ("Floor" == simple_type)
            return parseFloorAttr(op_info, op_attr);
        else if ("Cast" == simple_type)
            return parseCastAttr(op_info, op_attr);
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported simple op type: {}", simple_type);
        return false;
    }

    Operation* SimpleOperationsOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        SimpleOperationsOpAttr op_attr;
        std::string simple_type;
        if (!parseValue<std::string>(op_info, m_op_name, "simple_type", simple_type))
            return nullptr;
        if (!parseOpAttr(simple_type, op_info, op_attr))
            return nullptr;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, simple_type);
        if ("DataConvert" == simple_type)
        {
            return graph->CreateOperation<ops::DataConvert>().get();
        }
        else if ("Neg" == simple_type)
        {
            return graph->CreateOperation<ops::Neg>().get();
        }
        else if ("Abs" == simple_type)
        {
            return graph->CreateOperation<ops::Abs>().get();
        }
        else if ("Sin" == simple_type)
        {
            return graph->CreateOperation<ops::Sin>().get();
        }
        else if ("Exp" == simple_type)
        {
            return graph->CreateOperation<ops::Exp>().get();
        }
        else if ("Log" == simple_type)
        {
            return graph->CreateOperation<ops::Log>().get();
        }
        else if ("Sqrt" == simple_type)
        {
            return graph->CreateOperation<ops::Sqrt>().get();
        }
        else if ("Rsqrt" == simple_type)
        {
            return graph->CreateOperation<ops::Rsqrt>().get();
        }
        else if ("Square" == simple_type)
        {
            return graph->CreateOperation<ops::Square>().get();
        }
        else if ("LogicalNot" == simple_type)
        {
            return graph->CreateOperation<ops::LogicalNot>().get();
        }
        else if ("Floor" == simple_type)
        {
            return graph->CreateOperation<ops::Floor>().get();
        }
        else if ("Cast" == simple_type)
        {
            return graph->CreateOperation<ops::Cast>().get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported simple op type: {}", simple_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(SimpleOperationsOpCreator, SimpleOperations);

} // namespace TimVX