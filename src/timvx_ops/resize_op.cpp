/***********************************
******  resize_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/resize.h"
#include "resize_op.h"


namespace TIMVXPY
{
    bool ResizeCreator::parseOpAttr(const json &op_info, ResizeOpAttr &op_attr)
    {
        op_attr.layout = DataLayout::WHCN;
        return parseResizeType(op_info, m_op_name, "type", op_attr.type)
            && parseValue<float>(op_info, m_op_name, "factor", op_attr.factor)
            && parseValue<bool>(op_info, m_op_name, "align_corners", op_attr.align_corners)
            && parseValue<bool>(op_info, m_op_name, "half_pixel_centers", op_attr.half_pixel_centers)
            && parseValue<int32_t>(op_info, m_op_name, "target_height", op_attr.target_height)
            && parseValue<int32_t>(op_info, m_op_name, "target_width", op_attr.target_width)
            && parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout);
    }

    Operation* ResizeCreator::onCreate(std::shared_ptr<Graph> &graph, const json &op_info)
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
        return graph->CreateOperation<ops::Resize>(type, factor, align_corners,
            half_pixel_centers, target_height, target_width, layout).get();
    }

    REGISTER_OP_CREATOR(ResizeCreator, Resize);
} // namespace TIMVXPY