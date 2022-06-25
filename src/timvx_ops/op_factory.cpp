/********************************************
 * @Author: zjd
 * @Date: 2022-05-23 
 * @LastEditTime: 2022-05-23 
 * @LastEditors: zjd
 ********************************************/
#include "op_factory.h"

namespace TIMVX 
{

    Operation* ModelFactory::create(const std::string &op_type, const json &op_info, std::shared_ptr<Graph> &graph)
    {
        auto creator = getOpCreator(op_type);
        if (nullptr == creator)
        {
            TIMVX_ERROR("currently not support %s creator\n", op_type.c_str());
            return nullptr;
        }
        Operation* op = creator->onCreate(graph, op_info);
        return op;
    }

} // namespace TIMVX
