/***********************************
******  maxunpool2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class MaxUnpool2dOpCreator : public OpCreator
    {
    public:
        struct MaxUnpool2dOpAttr
        {
            std::array<uint32_t, 2> ksize;
            std::array<uint32_t, 2> stride;
            DataLayout              layout;
        };

        MaxUnpool2dOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseKsizeAttr(const json& op_info, MaxUnpool2dOpAttr& op_attr);
        bool parseStrideAttr(const json& op_info, MaxUnpool2dOpAttr& op_attr);
        bool parseLayoutAttr(const json& op_info, MaxUnpool2dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, MaxUnpool2dOpAttr& op_attr);

    };

} // namespace TimVX