/***********************************
******  reduce_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/reduce.h"
#include "timvx_ops/reduce_op.h"

namespace TimVX
{

    bool ReduceOpCreator::parseAxisAttr(const json& op_info, ReduceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    bool ReduceOpCreator::parseKeepDimsAttr(const json& op_info, ReduceOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "keep_dims", op_attr.keep_dims);
    }

    bool ReduceOpCreator::parseOpAttr(const json& op_info, ReduceOpAttr& op_attr)
    {
        return parseAxisAttr(op_info, op_attr) && parseKeepDimsAttr(op_info, op_attr);
    }

    Operation* ReduceOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        std::string reduce_type;
        if (!parseValue<std::string>(op_info, m_op_name, "reduce_type", reduce_type))
            return nullptr;
        ReduceOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> axis = op_attr.axis;
        bool keep_dims            = op_attr.keep_dims;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, reduce_type);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, keep_dims);
        if ("Min" == reduce_type)
        {
            return graph->CreateOperation<ops::ReduceMin>(axis, keep_dims).get();
        }
        else if ("Max" == reduce_type)
        {
            return graph->CreateOperation<ops::ReduceMax>(axis, keep_dims).get();
        }
        else if ("Any" == reduce_type)
        {
            return graph->CreateOperation<ops::ReduceAny>(axis, keep_dims).get();
        }
        else if ("All" == reduce_type)
        {
            return graph->CreateOperation<ops::ReduceAll>(axis, keep_dims).get();
        }
        else if ("Prod" == reduce_type)
        {
            return graph->CreateOperation<ops::ReduceProd>(axis, keep_dims).get();
        }
        else if ("Mean" == reduce_type)
        {
            return graph->CreateOperation<ops::ReduceMean>(axis, keep_dims).get();
        }
        else if ("Sum" == reduce_type)
        {
            return graph->CreateOperation<ops::ReduceSum>(axis, keep_dims).get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported reduce_type op type: {}", reduce_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(ReduceOpCreator, Reduce);

} // namespace TimVX