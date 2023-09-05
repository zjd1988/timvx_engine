/***********************************
******  resize_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ResizeOpCreator : public OpCreator
    {
    public:
        struct ResizeOpAttr
        {
            ResizeType type;
            float      factor;
            bool       align_corners;
            bool       half_pixel_centers;
            int        target_height;
            int        target_width;
            DataLayout layout;
        };

        ResizeOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseTypeAttr(const json& op_info, ResizeOpAttr& op_attr);
        bool parseFactorAttr(const json& op_info, ResizeOpAttr& op_attr);
        bool parseAlignCornersAttr(const json& op_info, ResizeOpAttr& op_attr);
        bool parseHalfPixelCentersAttr(const json& op_info, ResizeOpAttr& op_attr);
        bool parseTargetHeightAttr(const json& op_info, ResizeOpAttr& op_attr);
        bool parseTargetWidthAttr(const json& op_info, ResizeOpAttr& op_attr);
        bool parseLayoutAttr(const json& op_info, ResizeOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, ResizeOpAttr& op_attr);

    };

} // namespace TimVX
