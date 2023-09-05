/***********************************
******  dropout_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class DropoutOpCreator : public OpCreator
    {
    public:
        struct DropoutOpAttr
        {
            float ratio;
        };

        DropoutOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseRatioAttr(const json& op_info, DropoutOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, DropoutOpAttr& op_attr);

    };

} // namespace TimVX
