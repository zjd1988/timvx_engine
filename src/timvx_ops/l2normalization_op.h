/***********************************
******  l2normalization_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class L2NormalizationOpCreator : public OpCreator
    {
    public:
        struct L2NormalizationOpAttr
        {
            int32_t axis;
        };

        L2NormalizationOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, L2NormalizationOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, L2NormalizationOpAttr& op_attr);

    };

} // namespace TimVX
