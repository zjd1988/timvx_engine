/***********************************
******  activation_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TIMVXPY
{

    class ActivationCreator : public OpCreator
    {
    public:
        struct ActivationOpAttr
        {
            // prelu parameter
            struct
            {
                int axis;
            }prelu;
            // leakyrelu parameter
            struct
            {
                float ratio = 1.0f;
            }leakyrelu;        
            // linear parameter
            struct
            {
                float a = 1.0f;
                float b = 0.0f;
            }linear;
            // gelu parameter
            struct
            {
                bool approximate = true;
            }gelu;
            // hard sigmoid parameter
            struct
            {
                float alpha;
                float beta;
            }hardsigmoid;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        bool parsePreluAttr(const json &op_info, ActivationOpAttr &op_attr);
        bool parseLeakyreluAttr(const json &op_info, ActivationOpAttr &op_attr);
        bool parseLinearAttr(const json &op_info, ActivationOpAttr &op_attr);
        bool parseGeluAttr(const json &op_info, ActivationOpAttr &op_attr);
        bool parseHardsigmoidAttr(const json &op_info, ActivationOpAttr &op_attr);
        bool parseOpAttr(std::string op_type, const json &op_info, ActivationOpAttr &op_attr);
    
    private:
        std::string m_op_name = "Activation";
    };

} // namespace TIMVXPY
