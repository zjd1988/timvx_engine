/********************************************
// Filename: io_util.cpp
// Created by zhaojiadi on 2021/5/28
// Description: 
// 
********************************************/
#include <fstream>
#include "common/timvx_log.h"
#include "common/io_uitl.h"

namespace TimVX
{

    int readFileData(std::string file_name, std::shared_ptr<char>& file_data, int& file_len)
    {
        file_data.reset();
        std::ifstream file_stream(file_name, std::ios::binary|std::ios::in);
        if (!file_stream.is_open())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "get file data from {} fail", file_name.c_str());
            return -1;
        }
        file_stream.seekg(0,std::ios::end);
        file_len = file_stream.tellg();
        file_stream.seekg(0,std::ios::beg);
        file_data.reset(new char[file_len], std::default_delete<char []>());
        int read_len = 0;
        file_stream.read(file_data.get(), file_len);
        if(!file_stream.bad()) 
        {
            read_len = file_stream.gcount();
        }
        if (read_len != file_len)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "read file {} bytes {} not equal to file actual bytes {}", 
                file_name.c_str(), read_len, file_len);
            return -1;
        }
        return 0;
    }

    int saveFileData(const std::string& file_name, const char* save_data, const size_t save_len)
    {
        if (nullptr == save_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "data to wirte is nullptr, please check", file_name);
            return -1;
        }
        if (0 >= save_len)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "data to wirte {} bytes, please check", save_len);
            return -1;
        }
        std::ofstream file_stream(file_name.c_str(), std::ios_base::binary);
        if (!file_stream.is_open())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "open file {} to wirte fail", file_name);
            return -1;
        }
        file_stream.write(save_data, save_len);
        if (file_stream.bad())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "write {} bytes to file {} fail, please check",
                save_len, file_name);
            return -1;
        }
        return 0;
    }

} // namespace TimVX 
