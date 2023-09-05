/***********************************
******  relational_operations_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class RelationalOperationsOpCreator : public OpCreator
    {
    public:
        struct RelationalOperationsOpAttr
        {
            // Greater parameter
            struct
            {
            } greater;
            // GreaterOrEqual parameter
            struct
            {
            } greater_or_equal;
            // Less parameter
            struct
            {
            } less;
            // LessOrEqual parameter
            struct
            {
            } less_or_equal;
            // NotEqual parameter
            struct
            {
            } not_equal;
            // Equal parameter
            struct
            {
            } equal;
        };

        RelationalOperationsOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseGreaterAttr(const json& op_info, RelationalOperationsOpAttr& op_attr);
        bool parseGreaterOrEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr);
        bool parseLessAttr(const json& op_info, RelationalOperationsOpAttr& op_attr);
        bool parseLessOrEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr);
        bool parseNotEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr);
        bool parseEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr);
        bool parseOpAttr(std::string relational_type, const json& op_info, RelationalOperationsOpAttr& op_attr);

    };

} // namespace TimVX
