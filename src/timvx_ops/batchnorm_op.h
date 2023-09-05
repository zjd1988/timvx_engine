/***********************************
******  batchnorm_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class BatchNormOpCreator : public OpCreator
    {
    public:
        struct BatchNormOpAttr
        {
            float eps;
        };

        BatchNormOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseEpsAttr(const json& op_info, BatchNormOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, BatchNormOpAttr& op_attr);

    };

} // namespace TimVX
