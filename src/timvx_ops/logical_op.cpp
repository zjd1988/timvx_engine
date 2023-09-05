/***********************************
******  logical_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/logical.h"
#include "timvx_ops/logical_op.h"

namespace TimVX
{

    bool LogicalOpCreator::parseOpAttr(std::string op_type, const json& op_info, LogicalOpAttr& op_attr)
    {
        return true;
    }

    Operation* LogicalOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        LogicalOpAttr op_attr;
        std::string logical_type;
        if (!parseValue<std::string>(op_info, m_op_name, "logical_type", logical_type))
            return nullptr;
        if (!parseOpAttr(logical_type, op_info, op_attr))
            return nullptr;
        
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, logical_type);
        if ("And" == logical_type)
        {
            return graph->CreateOperation<ops::LogicalAnd>().get();
        }
        else if ("Or" == logical_type)
        {
            return graph->CreateOperation<ops::LogicalOr>().get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported logical op type: {}", logical_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(LogicalOpCreator, Logical);

} // namespace TimVX