/***********************************
******  resize1d_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class Resize1dOpCreator : public OpCreator
    {
    public:
        struct Resize1dOpAttr
        {
            ResizeType type;
            float      factor;
            bool       align_corners;
            bool       half_pixel_centers;
            int        target_size;
            DataLayout layout;
        };

        Resize1dOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseTypeAttr(const json& op_info, Resize1dOpAttr& op_attr);
        bool parseFactorAttr(const json& op_info, Resize1dOpAttr& op_attr);
        bool parseAlignCornersAttr(const json& op_info, Resize1dOpAttr& op_attr);
        bool parseHalfPixelCentersAttr(const json& op_info, Resize1dOpAttr& op_attr);
        bool parseTargetSizeAttr(const json& op_info, Resize1dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, Resize1dOpAttr& op_attr);

    };

} // namespace TimVX
