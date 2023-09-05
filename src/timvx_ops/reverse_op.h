/***********************************
******  reverse_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ReverseOpCreator : public OpCreator
    {
    public:
        struct ReverseOpAttr
        {
            std::vector<int32_t> axis;
        };

        ReverseOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, ReverseOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, ReverseOpAttr& op_attr);

    };

} // namespace TimVX