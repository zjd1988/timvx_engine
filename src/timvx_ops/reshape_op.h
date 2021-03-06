/***********************************
******  reshape_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TIMVX
{

    class ReshapeCreator : public OpCreator
    {
    public:
        struct ReshapeOpAttr
        {
            std::vector<uint32_t> size;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        bool parseOpAttr(const json &op_info, ReshapeOpAttr &op_attr);

    private:
        std::string m_op_name = "Reshape";
    };

} // namespace TIMVX
