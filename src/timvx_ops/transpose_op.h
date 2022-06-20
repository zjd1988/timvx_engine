/***********************************
******  transpose_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{

    class TransposeCreator : public OpCreator
    {
    public:
        struct TransposeOpAttr
        {
            std::vector<uint32_t> perm;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        bool parseOpAttr(const json &op_info, TransposeOpAttr &op_attr);

    private:
        std::string m_op_name = "Transpose";
    };

} // namespace TIMVXPY
