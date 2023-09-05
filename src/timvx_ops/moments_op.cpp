/***********************************
******  moments_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/moments.h"
#include "timvx_ops/moments_op.h"

namespace TimVX
{

    bool MomentsOpCreator::parseAxesAttr(const json& op_info, MomentsOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "axes", op_attr.axes);
    }

    bool MomentsOpCreator::parseKeepDimsAttr(const json& op_info, MomentsOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "keep_dims", op_attr.keep_dims, false);
    }

    bool MomentsOpCreator::parseOpAttr(const json& op_info, MomentsOpAttr& op_attr)
    {
        return parseAxesAttr(op_info, op_attr) && parseKeepDimsAttr(op_info, op_attr);
    }

    Operation* MomentsOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        MomentsOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> axes = op_attr.axes;
        bool keep_dims            = op_attr.keep_dims;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axes);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, keep_dims);
        return graph->CreateOperation<ops::Moments>(axes, keep_dims).get();
    }

    REGISTER_OP_CREATOR(MomentsOpCreator, Moments);

} // namespace TimVX