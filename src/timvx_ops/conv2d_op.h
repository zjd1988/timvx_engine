/***********************************
******  conv2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TIMVXPY
{
    
    class Conv2dCreator : public OpCreator
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

        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        bool getConv2dType();
        bool parseWeights(const json &op_info, Conv2dOpAttr &op_attr);
        bool parsePadding(const json &op_info, Conv2dOpAttr &op_attr);
        bool parseKsize(const json &op_info, Conv2dOpAttr &op_attr);
        bool parseStride(const json &op_info, Conv2dOpAttr &op_attr);
        bool parseDilation(const json &op_info, Conv2dOpAttr &op_attr);
        bool parsePad(const json &op_info, Conv2dOpAttr &op_attr);
        bool parseMultiplier(const json &op_info, Conv2dOpAttr &op_attr);
        bool parseInputLayout(const json &op_info, Conv2dOpAttr &op_attr);
        bool parseKernelLayout(const json &op_info, Conv2dOpAttr &op_attr);
        bool parseOpAttr(const json &op_info, Conv2dOpAttr &op_attr);

    private:
        std::string m_op_name = "Conv2d";
    };

} // namespace TIMVXPY
