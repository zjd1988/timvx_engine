/***********************************
******  softmax_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class SoftmaxOpCreator : public OpCreator
    {
    public:
        struct SoftmaxOpAttr
        {
            float   beta;
            int32_t axis;
        };

        SoftmaxOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseBetaAttr(const json& op_info, SoftmaxOpAttr& op_attr);
        bool parseAxisAttr(const json& op_info, SoftmaxOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, SoftmaxOpAttr& op_attr);

    };

} // namespace TimVX
