/***********************************
******  stridedslice_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/stridedslice.h"
#include "timvx_ops/stridedslice_op.h"

namespace TimVX
{

    bool StridedSliceOpCreator::parseBeginDimsAttr(const json& op_info, StridedSliceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "begin_dims", op_attr.begin_dims);
    }

    bool StridedSliceOpCreator::parseEndDimsAttr(const json& op_info, StridedSliceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "end_dims", op_attr.end_dims);
    }

    bool StridedSliceOpCreator::parseStrideDimsAttr(const json& op_info, StridedSliceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "stride_dims", op_attr.stride_dims);
    }

    bool StridedSliceOpCreator::parseBeginMaskAttr(const json& op_info, StridedSliceOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "begin_mask", op_attr.begin_mask);
    }

    bool StridedSliceOpCreator::parseEndMaskAttr(const json& op_info, StridedSliceOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "end_mask", op_attr.end_mask);
    }

    bool StridedSliceOpCreator::parseShrinkAxisMaskAttr(const json& op_info, StridedSliceOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "shrink_axis_mask", op_attr.shrink_axis_mask);
    }

    bool StridedSliceOpCreator::parseOpAttr(const json& op_info, StridedSliceOpAttr& op_attr)
    {
        return parseBeginDimsAttr(op_info, op_attr) && parseEndDimsAttr(op_info, op_attr) && 
            parseStrideDimsAttr(op_info, op_attr) && parseBeginMaskAttr(op_info, op_attr) && 
            parseEndMaskAttr(op_info, op_attr) && parseShrinkAxisMaskAttr(op_info, op_attr);
    }

    Operation* StridedSliceOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        StridedSliceOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> begin_dims  = op_attr.begin_dims;
        std::vector<int32_t> end_dims    = op_attr.end_dims;
        std::vector<int32_t> stride_dims = op_attr.stride_dims;
        int32_t begin_mask               = op_attr.begin_mask;
        int32_t end_mask                 = op_attr.end_mask;
        int32_t shrink_axis_mask         = op_attr.shrink_axis_mask;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, begin_dims);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, end_dims);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, stride_dims);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, begin_mask);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, end_mask);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, shrink_axis_mask);
        return graph->CreateOperation<ops::StridedSlice>(begin_dims, end_dims, stride_dims, 
            begin_mask, end_mask, shrink_axis_mask).get();
    }

    REGISTER_OP_CREATOR(StridedSliceOpCreator, StridedSlice);

} // namespace TimVX