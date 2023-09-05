/***********************************
******  eltwise_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class EltwiseOpCreator : public OpCreator
    {
    public:
        struct EltwiseOpAttr
        {
        };

        EltwiseOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    };

} // namespace TimVX
