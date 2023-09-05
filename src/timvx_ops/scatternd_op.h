/***********************************
******  scatternd_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ScatterNDOpCreator : public OpCreator
    {
    public:
        struct ScatterNDOpAttr
        {
            std::vector<uint32_t> shape;
        };

        ScatterNDOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseShapeAttr(const json& op_info, ScatterNDOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, ScatterNDOpAttr& op_attr);

    };

} // namespace TimVX