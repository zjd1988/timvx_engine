/***********************************
******  reduce_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ReduceOpCreator : public OpCreator
    {
    public:
        struct ReduceOpAttr
        {
            std::vector<int32_t> axis;
            bool                 keep_dims;
        };

        ReduceOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, ReduceOpAttr& op_attr);
        bool parseKeepDimsAttr(const json& op_info, ReduceOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, ReduceOpAttr& op_attr);

    };

} // namespace TimVX