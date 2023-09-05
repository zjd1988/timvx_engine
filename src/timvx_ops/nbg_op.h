/***********************************
******  nbg_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class NBGOpCreator : public OpCreator
    {
    public:
        struct NBGOpAttr
        {
            void* binary;
            size_t input_count;
            size_t output_count;
        };

        NBGOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseBinaryAttr(const json& op_info, NBGOpAttr& op_attr);
        bool parseInputCountAttr(const json& op_info, NBGOpAttr& op_attr);
        bool parseOutputCountAttr(const json& op_info, NBGOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, NBGOpAttr& op_attr);

    };

} // namespace TimVX
