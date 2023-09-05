/***********************************
******  matmul_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class MatmulOpCreator : public OpCreator
    {
    public:
        struct MatmulOpAttr
        {
            bool transpose_a;
            bool transpose_b;
            bool adjoint_a;
            bool adjoint_b;
        };

        MatmulOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseTransposeAAttr(const json& op_info, MatmulOpAttr& op_attr);
        bool parseTransposeBAttr(const json& op_info, MatmulOpAttr& op_attr);
        bool parseAdjointAAttr(const json& op_info, MatmulOpAttr& op_attr);
        bool parseAdjointBAttr(const json& op_info, MatmulOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, MatmulOpAttr& op_attr);

    };

} // namespace TimVX
