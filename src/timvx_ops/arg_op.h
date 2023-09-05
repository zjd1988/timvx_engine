/***********************************
******  arg_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ArgOpCreator : public OpCreator
    {
    public:
        struct ArgOpAttr
        {
            // max parameter
            struct
            {
                int32_t axis;
            } max;
            // min parameter
            struct
            {
                int32_t axis;
            } min;
        };

        ArgOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseMaxAttr(const json& op_info, ArgOpAttr& op_attr);
        bool parseMinAttr(const json& op_info, ArgOpAttr& op_attr);
        bool parseOpAttr(std::string op_type, const json& op_info, ArgOpAttr& op_attr);

    };

} // namespace TimVX
