/***********************************
******  split_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class SplitOpCreator : public OpCreator
    {
    public:
        struct SplitOpAttr
        {
            uint32_t              axis;
            std::vector<uint32_t> slices;
        };

        SplitOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, SplitOpAttr& op_attr);
        bool parseSlicesAttr(const json& op_info, SplitOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, SplitOpAttr& op_attr);

    };

} // namespace TimVX