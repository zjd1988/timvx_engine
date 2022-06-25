/********************************************
 * @Author: zjd
 * @Date: 2022-05-23 
 * @LastEditTime: 2022-05-23 
 * @LastEditors: zjd
 ********************************************/
#pragma once
#include "op_creator.h"
#include "timvx_define.h"
#include "nlohmann/json.hpp"
using namespace nlohmann;

namespace TIMVX 
{

    /** operator factory */
    class OpFactory
    {
    public:
        /**
         * @brief create op with given info.
         * @param op_type  operator type.
         * @param op_info  operator info.
         * @param graph    timvx graph.
         * @return created op or NULL if failed.
         */
        static Operation* create(const string &op_type, const json &op_info, std::shared_ptr<Graph> &graph);
    };

} // namespace TIMVX

