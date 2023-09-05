/***********************************
******  io_util.h
******
******  Created by zhaojd on 2022/05/04.
***********************************/
#pragma once
#include <memory>
#include <iostream>

namespace TimVX
{

    int readFileData(std::string file_name, std::shared_ptr<char>& file_data, int& file_len);

} // namespace TimVX