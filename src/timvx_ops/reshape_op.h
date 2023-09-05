/***********************************
******  reshape_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ReshapeOpCreator : public OpCreator
    {
    public:
        struct ReshapeOpAttr
        {
            std::vector<uint32_t> size;
        };

        ReshapeOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseSizeAttr(const json& op_info, ReshapeOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, ReshapeOpAttr& op_attr);

    };

} // namespace TimVX
