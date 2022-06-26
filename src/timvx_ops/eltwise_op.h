/***********************************
******  eltwise_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TIMVX
{

    class EltwiseCreator : public OpCreator
    {
    public:
        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        std::string m_op_name = "Eltwise";
    };

} // namespace TIMVX
