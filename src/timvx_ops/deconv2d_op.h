/***********************************
******  deconv2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{
    
    class DeConv2dOpCreator : public OpCreator
    {
    public:
        struct DeConv2dOpAttr
        {
            uint32_t                oc_count;
            PadType                 pad_type;
            std::array<uint32_t, 2> ksize;
            std::array<uint32_t, 2> stride;
            std::array<uint32_t, 2> output_padding;
            std::array<uint32_t, 4> pad;
            uint32_t                group;
            DataLayout              input_layout;
            DataLayout              kernel_layout;
        };

        DeConv2dOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseOcCountAttr(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parsePaddingAttr(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseKsizeAttr(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseStrideAttr(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseOutputPaddingAttr(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parsePadAttr(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseGroupAttr(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseKernelLayoutAttr(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, DeConv2dOpAttr& op_attr);

    };

} // namespace TimVX
