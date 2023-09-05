/***********************************
******  pad_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class PadOpCreator : public OpCreator
    {
    public:
        struct PadOpAttr
        {
            std::vector<uint32_t> front_size;
            std::vector<uint32_t> back_size;
            int32_t               const_val;
        };

        PadOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseFrontSizeAttr(const json& op_info, PadOpAttr& op_attr);
        bool parseBackSizeAttr(const json& op_info, PadOpAttr& op_attr);
        bool parseConstValAttr(const json& op_info, PadOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, PadOpAttr& op_attr);

    };

} // namespace TimVX