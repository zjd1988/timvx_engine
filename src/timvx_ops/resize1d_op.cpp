/***********************************
******  resize1d_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/resize1d.h"
#include "timvx_ops/resize1d_op.h"

namespace TimVX
{

    bool Resize1dOpCreator::parseTypeAttr(const json& op_info, Resize1dOpAttr& op_attr)
    {
        return parseResizeType(op_info, m_op_name, "type", op_attr.type);
    }

    bool Resize1dOpCreator::parseFactorAttr(const json& op_info, Resize1dOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "factor", op_attr.factor);
    }

    bool Resize1dOpCreator::parseAlignCornersAttr(const json& op_info, Resize1dOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "align_corners", op_attr.align_corners);
    }

    bool Resize1dOpCreator::parseHalfPixelCentersAttr(const json& op_info, Resize1dOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "half_pixel_centers", op_attr.half_pixel_centers);
    }

    bool Resize1dOpCreator::parseTargetSizeAttr(const json& op_info, Resize1dOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "target_size", op_attr.target_size);
    }

    bool Resize1dOpCreator::parseOpAttr(const json& op_info, Resize1dOpAttr& op_attr)
    {
        op_attr.layout = DataLayout::WHCN;
        return parseTypeAttr(op_info, op_attr) && parseFactorAttr(op_info, op_attr) && 
            parseAlignCornersAttr(op_info, op_attr) && parseHalfPixelCentersAttr(op_info, op_attr) && 
            parseTargetSizeAttr(op_info, op_attr);
    }

    Operation* Resize1dOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        Resize1dOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        ResizeType type         = op_attr.type;
        float factor            = op_attr.factor;
        bool align_corners      = op_attr.align_corners;
        bool half_pixel_centers = op_attr.half_pixel_centers;
        int target_size         = op_attr.target_size;
        DataLayout layout       = op_attr.layout;

        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, type, gResizeTypeToStrMap[type]);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, factor);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, align_corners);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, half_pixel_centers);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, target_size);
        return graph->CreateOperation<ops::Resize1d>(type, factor, align_corners,
            half_pixel_centers, target_size, layout).get();
    }

    REGISTER_OP_CREATOR(Resize1dOpCreator, Resize1d);

} // namespace TimVX