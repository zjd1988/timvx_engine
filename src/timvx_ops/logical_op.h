/***********************************
******  logical_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class LogicalOpCreator : public OpCreator
    {
    public:
        struct LogicalOpAttr
        {
            // and parameter
            struct
            {
            } and_attr;
            // or parameter
            struct
            {
            } or_attr;
        };

        LogicalOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseOpAttr(std::string op_type, const json& op_info, LogicalOpAttr& op_attr);

    };

} // namespace TimVX
