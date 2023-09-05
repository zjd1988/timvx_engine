/***********************************
******  stack_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class StackOpCreator : public OpCreator
    {
    public:
        struct StackOpAttr
        {
            uint32_t axis;
            int32_t  input_cnt;
        };

        StackOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, StackOpAttr& op_attr);
        bool parseInputCntAttr(const json& op_info, StackOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, StackOpAttr& op_attr);

    };

} // namespace TimVX