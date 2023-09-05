/***********************************
******  resize_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/resize.h"
#include "timvx_ops/resize_op.h"

namespace TimVX
{

    bool ResizeOpCreator::parseTypeAttr(const json& op_info, ResizeOpAttr& op_attr)
    {
        return parseResizeType(op_info, m_op_name, "type", op_attr.type);
    }

    bool ResizeOpCreator::parseFactorAttr(const json& op_info, ResizeOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "factor", op_attr.factor);
    }

    bool ResizeOpCreator::parseAlignCornersAttr(const json& op_info, ResizeOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "align_corners", op_attr.align_corners);
    }

    bool ResizeOpCreator::parseHalfPixelCentersAttr(const json& op_info, ResizeOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "half_pixel_centers", op_attr.half_pixel_centers);
    }

    bool ResizeOpCreator::parseTargetHeightAttr(const json& op_info, ResizeOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "target_height", op_attr.target_height);
    }

    bool ResizeOpCreator::parseTargetWidthAttr(const json& op_info, ResizeOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "target_width", op_attr.target_width);
    }

    bool ResizeOpCreator::parseLayoutAttr(const json& op_info, ResizeOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout);
    }

    bool ResizeOpCreator::parseOpAttr(const json& op_info, ResizeOpAttr& op_attr)
    {
        op_attr.layout = DataLayout::WHCN;
        return parseTypeAttr(op_info, op_attr) && parseFactorAttr(op_info, op_attr) && 
            parseAlignCornersAttr(op_info, op_attr) && parseHalfPixelCentersAttr(op_info, op_attr) && 
            parseTargetHeightAttr(op_info, op_attr) && parseTargetWidthAttr(op_info, op_attr) && 
            parseLayoutAttr(op_info, op_attr);
    }

    Operation* ResizeOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ResizeOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        ResizeType type         = op_attr.type;
        float factor            = op_attr.factor;
        bool align_corners      = op_attr.align_corners;
        bool half_pixel_centers = op_attr.half_pixel_centers;
        int target_height       = op_attr.target_height;
        int target_width        = op_attr.target_width;
        DataLayout layout       = op_attr.layout;

        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, type, gResizeTypeToStrMap[type]);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, factor);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, align_corners);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, half_pixel_centers);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, target_height);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, target_width);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, layout, gDataLayoutToStrMap[layout]);
        return graph->CreateOperation<ops::Resize>(type, factor, align_corners,
            half_pixel_centers, target_height, target_width, layout).get();
    }

    REGISTER_OP_CREATOR(ResizeOpCreator, Resize);

} // namespace TimVX