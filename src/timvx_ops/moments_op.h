/***********************************
******  moments_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class MomentsOpCreator : public OpCreator
    {
    public:
        struct MomentsOpAttr
        {
            std::vector<int32_t> axes;
            bool                 keep_dims;
        };

        MomentsOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxesAttr(const json& op_info, MomentsOpAttr& op_attr);
        bool parseKeepDimsAttr(const json& op_info, MomentsOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, MomentsOpAttr& op_attr);

    };

} // namespace TimVX