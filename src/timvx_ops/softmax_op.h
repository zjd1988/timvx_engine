/***********************************
******  softmax_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TIMVX
{

    class SoftmaxCreator : public OpCreator
    {
    public:
        struct SoftmaxOpAttr
        {
            float beta;
            int32_t axis;
        };
    
        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        bool parseOpAttr(const json &op_info, SoftmaxOpAttr &op_attr);

    private:
        std::string m_op_name = "Softmax";
    };

} // namespace TIMVX
