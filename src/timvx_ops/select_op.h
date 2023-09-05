/***********************************
******  select_op.h
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class SelectOpCreator : public OpCreator
    {
    public:
        struct SelectOpAttr
        {
        };

        SelectOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseOpAttr(const json& op_info, SelectOpAttr& op_attr);

    };

} // namespace TimVX
