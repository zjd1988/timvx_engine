/***********************************
******  spatial_transformer_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class SpatialTransformerOpCreator : public OpCreator
    {
    public:
        struct SpatialTransformerOpAttr
        {
            uint32_t output_h;
            uint32_t output_w;
            bool     has_theta_1_1;
            bool     has_theta_1_2;
            bool     has_theta_1_3;
            bool     has_theta_2_1;
            bool     has_theta_2_2;
            bool     has_theta_2_3;
            float    theta_1_1;
            float    theta_1_2;
            float    theta_1_3;
            float    theta_2_1;
            float    theta_2_2;
            float    theta_2_3;
        };

        SpatialTransformerOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseOutputHAttr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseOutputWAttr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseHasTheta11Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseHasTheta12Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseHasTheta13Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseHasTheta21Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseHasTheta22Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseHasTheta23Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseTheta11Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseTheta12Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseTheta13Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseTheta21Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseTheta22Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseTheta23Attr(const json& op_info, SpatialTransformerOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, SpatialTransformerOpAttr& op_attr);

    };

} // namespace TimVX