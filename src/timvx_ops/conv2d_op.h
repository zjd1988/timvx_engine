/***********************************
******  conv2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{
    
    class Conv2dOpCreator : public OpCreator
    {
    public:
        struct Conv2dOpAttr
        {
            uint32_t                weights;
            PadType                 padding;
            std::array<uint32_t, 2> ksize;
            std::array<uint32_t, 2> stride;
            std::array<uint32_t, 2> dilation;
            std::array<uint32_t, 4> pad;
            int32_t                 multiplier;
            DataLayout              input_layout;
            DataLayout              kernel_layout;
        };

        Conv2dOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseWeightsAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parsePaddingAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parseKsizeAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parseStrideAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parseDilationAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parsePadAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parseMultiplierAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parseInputLayoutAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parseKernelLayoutAttr(const json& op_info, Conv2dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, Conv2dOpAttr& op_attr);

    };

} // namespace TimVX
