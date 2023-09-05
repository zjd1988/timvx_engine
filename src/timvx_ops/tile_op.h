/***********************************
******  tile_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class TileOpCreator : public OpCreator
    {
    public:
        struct TileOpAttr
        {
            std::vector<int32_t> multiples;
        };

        TileOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseMultiplesAttr(const json& op_info, TileOpAttr& TileOpAttr);
        bool parseOpAttr(const json& op_info, TileOpAttr& op_attr);

    };

} // namespace TimVX