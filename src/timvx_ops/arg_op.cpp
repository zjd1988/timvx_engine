/***********************************
******  arg_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/arg.h"
#include "timvx_ops/arg_op.h"

namespace TimVX
{

    bool ArgOpCreator::parseMaxAttr(const json& op_info, ArgOpAttr& op_attr)
    {
        std::string full_name = m_op_name + "Max";
        return parseValue<int>(op_info, full_name, "axis", op_attr.max.axis);
    }

    bool ArgOpCreator::parseMinAttr(const json& op_info, ArgOpAttr& op_attr)
    {
        std::string full_name = m_op_name + "Min";
        return parseValue<int>(op_info, full_name, "axis", op_attr.min.axis);
    }

    bool ArgOpCreator::parseOpAttr(std::string op_type, const json& op_info, ArgOpAttr& op_attr)
    {
        if (op_type == "Max")
            return parseMaxAttr(op_info, op_attr);
        else if (op_type == "Min")
            return parseMinAttr(op_info, op_attr);
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported arg op type: {}", op_type);
        return false;
    }

    Operation* ArgOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ArgOpAttr op_attr;
        std::string arg_type;
        if (!parseValue<std::string>(op_info, m_op_name, "arg_type", arg_type))
            return nullptr;
        if (!parseOpAttr(arg_type, op_info, op_attr))
            return nullptr;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, arg_type);
        if ("Max" == arg_type)
        {
            int axis = op_attr.max.axis;
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
            return graph->CreateOperation<ops::ArgMax>(axis).get();
        }
        else if ("Min" == arg_type)
        {
            int axis = op_attr.min.axis;
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
            return graph->CreateOperation<ops::ArgMin>(axis).get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported arg op type: {}", arg_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(ArgOpCreator, Arg);

} // namespace TimVX