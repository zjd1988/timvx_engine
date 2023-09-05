/***********************************
******  concat_op.h
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ConcatOpCreator : public OpCreator
    {
    public:
        struct ConcatOpAttr
        {
            uint32_t axis;
            int32_t  input_cnt;
        };

        ConcatOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, ConcatOpAttr& op_attr);
        bool parseInputCntAttr(const json& op_info, ConcatOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, ConcatOpAttr& op_attr);

    };

} // namespace TimVX
