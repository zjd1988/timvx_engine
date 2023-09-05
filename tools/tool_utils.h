/***********************************
******  tool_utils.h
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#pragma once
#include "cxxopts.hpp"
#include "common/non_copyable.h"
#include "timvx_c_api.h"

namespace TimVX
{

    typedef struct CmdLineArgOption
    {
        // model infer/compie
        std::string                    para_file;
        std::string                    weight_file;
        std::string                    input_file;
        bool                           output_flag;
        bool                           pass_through = false;
        bool                           is_prealloc = false;
        bool                           want_float = false;
        // model compile
        std::string                    compile_para_file;
        std::string                    compile_weight_file;
        // model benchmark
        int                            benchmark_times;
        // common
        std::string                    log_path;
        int                            log_level;
        bool                           help_flag;
    } CmdLineArgOption;

    class ModelTensorData : public NonCopyable
    {
    public:
        ModelTensorData(const char* file_name);
        ModelTensorData(std::vector<int> shape, TimvxTensorType type, 
            TimvxTensorFormat format, bool random_init=true);
        ModelTensorData(std::vector<int> shape, TimvxTensorType type, 
            TimvxTensorFormat format, void* data=nullptr);
        ~ModelTensorData();

    int tensorLength() { return m_data_len; };
    void* tensorData();
    int tensorElementCount();
    int tensorElementSize();
    TimvxTensorType tensorType() { return m_type; }
    TimvxTensorFormat tensorFormat() { return m_format; }

    // save data to npy file
    int saveDataToNpy(const char* file_name);

    private:
        // radom init data
        void randomInitData();
        // load data from image/npy
        int loadDataFromStb(const char* file_name);
        int loadDataFromNpy(const char* file_name);
    
    private:
        bool                           m_tensor_valid = false;
        bool                           m_own_flag = true;
        uint8_t*                       m_data = nullptr;
        int                            m_data_len = -1;
        TimvxTensorType                m_type;
        TimvxTensorFormat              m_format;
        std::vector<int>               m_shape;
    };

} // namespace TimVX
