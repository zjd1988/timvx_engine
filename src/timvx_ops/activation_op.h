/***********************************
******  activation_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ActivationOpCreator : public OpCreator
    {
    public:
        struct ActivationOpAttr
        {
            // relu parameter
            struct 
            {
            } relu;
            // relu1 parameter
            struct 
            {
            } relu1;
            // relu6 parameter
            struct 
            {
            } relu6;
            // elu parameter
            struct 
            {
            } elu;
            // tanh parameter
            struct 
            {
            } tanh;
            // sigmoid parameter
            struct 
            {
            } sigmoid;
            // hardswish parameter
            struct 
            {
            } hardswish;
            // mish parameter
            struct 
            {
            } mish;
            // hardsigmoid parameter
            struct 
            {
            } hardsigmoid;
            // softrelu parameter
            struct 
            {
            } softrelu;
            // prelu parameter
            struct
            {
                int32_t axis;
            } prelu;
            // leakyrelu parameter
            struct
            {
                float ratio = 1.0f;
            } leakyrelu;
            // linear parameter
            struct
            {
                float a = 1.0f;
                float b = 0.0f;
            } linear;
            // gelu parameter
            // struct
            // {
            //     bool approximate = true;
            // } gelu;
            // hard sigmoid parameter
            // struct
            // {
            //     float alpha;
            //     float beta;
            // } hardsigmoid;
        };

        ActivationOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseReluAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseRelu1Attr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseRelu6Attr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseEluAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseTanhAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseSigmoidAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseHardSwishAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseMishAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseHardSigmoidAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseSoftReluAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parsePreluAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseLeakyreluAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseLinearAttr(const json& op_info, ActivationOpAttr& op_attr);
        // bool parseGeluAttr(const json& op_info, ActivationOpAttr& op_attr);
        // bool parseHardsigmoidAttr(const json& op_info, ActivationOpAttr& op_attr);
        bool parseOpAttr(std::string op_type, const json& op_info, ActivationOpAttr& op_attr);
    
    };

} // namespace TimVX
