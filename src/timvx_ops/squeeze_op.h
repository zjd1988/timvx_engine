/***********************************
******  squeeze_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class SqueezeOpCreator : public OpCreator
    {
    public:
        struct SqueezeOpAttr
        {
            std::vector<uint32_t> axis;
        };

        SqueezeOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, SqueezeOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, SqueezeOpAttr& op_attr);

    };

} // namespace TimVX