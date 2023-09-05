/***********************************
******  deconv1d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class DeConv1dOpCreator : public OpCreator
    {
    public:
        struct DeConv1dOpAttr
        {
            uint32_t                oc_count; // output channel count
            PadType                 pad_type;
            uint32_t                ksize;
            uint32_t                stride;
            uint32_t                output_padding;
            std::array<uint32_t, 2> pad;
            uint32_t                group;
            DataLayout              input_layout;
            DataLayout              kernel_layout;
        };

        DeConv1dOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseOcCountAttr(const json& op_info, DeConv1dOpAttr& op_attr);
        bool parsePaddingAttr(const json& op_info, DeConv1dOpAttr& op_attr);
        bool parseKsizeAttr(const json& op_info, DeConv1dOpAttr& op_attr);
        bool parseStrideAttr(const json& op_info, DeConv1dOpAttr& op_attr);
        bool parseOutputPaddingAttr(const json& op_info, DeConv1dOpAttr& op_attr);
        bool parsePadAttr(const json& op_info, DeConv1dOpAttr& op_attr);
        bool parseGroupAttr(const json& op_info, DeConv1dOpAttr& op_attr);
        bool parseKernelLayoutAttr(const json& op_info, DeConv1dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, DeConv1dOpAttr& op_attr);

    };

} // namespace TimVX