/***********************************
******  transpose_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class TransposeOpCreator : public OpCreator
    {
    public:
        struct TransposeOpAttr
        {
            std::vector<uint32_t> perm;
        };

        TransposeOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parsePermAttr(const json& op_info, TransposeOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, TransposeOpAttr& op_attr);

    };

} // namespace TimVX
