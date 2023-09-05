/***********************************
******  unstack_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class UnstackOpCreator : public OpCreator
    {
    public:
        struct UnstackOpAttr
        {
            int32_t  axis;
            uint32_t output_num;
        };

        UnstackOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, UnstackOpAttr& op_attr);
        bool parseOutputNumAttr(const json& op_info, UnstackOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, UnstackOpAttr& op_attr);

    };

} // namespace TimVX