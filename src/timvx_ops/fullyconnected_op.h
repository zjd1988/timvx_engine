/***********************************
******  fullconnected_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class FullyConnectedOpCreator : public OpCreator
    {
    public:
        struct FullyConnectedOpAttr
        {
            uint32_t axis;
            uint32_t weights;
        };

        FullyConnectedOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, FullyConnectedOpAttr& op_attr);
        bool parseWeightsAttr(const json& op_info, FullyConnectedOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, FullyConnectedOpAttr& op_attr);

    };

} // namespace TimVX
