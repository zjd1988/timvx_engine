/***********************************
******  simple_operations_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class SimpleOperationsOpCreator : public OpCreator
    {
    public:
        struct SimpleOperationsOpAttr
        {
            // DataConvert parameter
            struct
            {
            } data_convert;
            // Neg parameter
            struct
            {
            } neg;
            // Abs parameter
            struct
            {
            } abs;
            // Sin parameter
            struct
            {
            } sin;
            // Exp parameter
            struct
            {
            } exp;
            // Log parameter
            struct
            {
            } log;
            // Sqrt parameter
            struct
            {
            } sqrt;
            // Rsqrt parameter
            struct
            {
            } rsqrt;
            // Square parameter
            struct
            {
            } square;
            // LogicalNot parameter
            struct
            {
            } logical_not;
            // Floor parameter
            struct
            {
            } floor;
            // Cast parameter
            struct
            {
            } cast;
        };

        SimpleOperationsOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseDataConvertAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseNegAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseAbsAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseSinAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseExpAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseLogAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseSqrtAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseRsqrtAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseSquareAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseLogicalNotAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseFloorAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseCastAttr(const json& op_info, SimpleOperationsOpAttr& op_attr);
        bool parseOpAttr(std::string simple_type, const json& op_info, SimpleOperationsOpAttr& op_attr);

    };

} // namespace TimVX
