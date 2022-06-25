/***********************************
******  fullconnected_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TIMVXPY
{

    class FullyConnectedCreator : public OpCreator
    {
    public:
        struct FullyConnectedOpAttr
        {
            uint32_t axis;
            uint32_t weights;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        bool parseOpAttr(const json &op_info, FullyConnectedOpAttr &op_attr);

    private:
        std::string m_op_name = "FullyConnected";
    };

} // namespace TIMVXPY
