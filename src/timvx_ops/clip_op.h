/***********************************
******  clip_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ClipOpCreator : public OpCreator
    {
    public:
        struct ClipOpAttr
        {
            float min;
            float max;
        };

        ClipOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseMinAttr(const json& op_info, ClipOpAttr& op_attr);
        bool parseMaxAttr(const json& op_info, ClipOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, ClipOpAttr& op_attr);

    };

} // namespace TimVX
