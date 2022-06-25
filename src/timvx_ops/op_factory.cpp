/********************************************
 * @Author: zjd
 * @Date: 2022-05-23 
 * @LastEditTime: 2022-05-23 
 * @LastEditors: zjd
 ********************************************/
#include "op_factory.h"

namespace TIMVX 
{

    Operation* ModelFactory::create(const string &op_type, const json &op_info, std::shared_ptr<Graph> &graph)
    {
        auto creator = getOpCreator(op_type);
        if (nullptr == creator)
        {
            std::cout << "currently not support " << op_type <<" creator" << std::endl;
            TIMVX_ERROR();
            return nullptr;
        }
        Operation* op = creator->onCreate(graph, op_info);
        return op;
    }

} // namespace TIMVX
