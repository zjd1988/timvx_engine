/***********************************
******  concat_op.h
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TIMVX
{

    class ConcatCreator : public OpCreator
    {
    public:
        struct ConcatOpAttr
        {
            uint32_t axis;
            int input_cnt;
        };
    
        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        bool parseOpAttr(const json &op_info, ConcatOpAttr &op_attr);

    private:
        std::string m_op_name = "Concat";
    };
} // namespace TIMVX
