/********************************************
// Filename: timvx_log.h
// Created by zhaojiadi on 2021/5/28
// Description: 
// 
********************************************/
/*
日志等级：trace, debug, info, warn, err, critical
使用方法：包含TIMVX_log.h头文件,调用初始化函数,使用TIMVX_Debug等打印日志信息
例：
TIMVXLog::Instance().InitTIMVXLog("scenario_edit", "scenario_edit_log.txt");
int i = 10;
double d_number = 10.01;
TIMVX_LOG(TIMVX_LEVEL_DEBUG, "TimVX log message");
TIMVX_LOG(TIMVX_LEVEL_DEBUG, "TimVX log message #{0}, d_number:{1}", i, d_number);
注：使用{}格式化字符串，里面的数字为占位符
https://github.com/gabime/spdlog

spdlog::info("Welcome to spdlog!");
spdlog::error("Some error message with arg: {}", 1);

spdlog::warn("Easy padding in numbers like {:08d}", 12);
spdlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
spdlog::info("Support for floats {:03.2f}", 1.23456);
spdlog::info("Positional args are {1} {0}..", "too", "supported");
spdlog::info("{:<30}", "left aligned");

spdlog::set_level(spdlog::level::debug); // Set global log level to debug
spdlog::debug("This message should be displayed..");

*/
#pragma once
#include <iostream>
#include <memory>
#include "spdlog/spdlog.h"
#include "common/non_copyable.h"

// #define SPDLOG_LEVEL_TRACE 0
// #define SPDLOG_LEVEL_DEBUG 1
// #define SPDLOG_LEVEL_INFO 2
// #define SPDLOG_LEVEL_WARN 3
// #define SPDLOG_LEVEL_ERROR 4
// #define SPDLOG_LEVEL_CRITICAL 5
// #define SPDLOG_LEVEL_OFF 6

#define TIMVX_LEVEL_TRACE    SPDLOG_LEVEL_TRACE
#define TIMVX_LEVEL_DEBUG    SPDLOG_LEVEL_DEBUG
#define TIMVX_LEVEL_INFO     SPDLOG_LEVEL_INFO
#define TIMVX_LEVEL_WARN     SPDLOG_LEVEL_WARN
#define TIMVX_LEVEL_ERROR    SPDLOG_LEVEL_ERROR
#define TIMVX_LEVEL_FATAL    SPDLOG_LEVEL_CRITICAL
#define TIMVX_LEVEL_OFF      SPDLOG_LEVEL_OFF


// #ifndef POSTFIX
// //log error str postfix with file name/func name/line num
// #define POSTFIX(msg) std::string(msg).append(" ")                            \
//                     .append(__FILE__).append("> <").append(__FUNCTION__)     \
//                     .append("> <").append(std::to_string(__LINE__))          \
//                     .append("")
// #endif //POSTFIX

// #define TIMVX_LOG_TRACE(...)  TIMVXLog::Instance().getLogger()->trace(__VA_ARGS__)
// #define TIMVX_LOG_DEBUG(...)       TIMVXLog::Instance().getLogger()->debug(__VA_ARGS__)
// #define TIMVX_LOG_INFO(...)        TIMVXLog::Instance().getLogger()->info(__VA_ARGS__)
// #define TIMVX_LOG_WARN(...)        TIMVXLog::Instance().getLogger()->warn(__VA_ARGS__)
// #define TIMVX_LOG_ERROR(...)  TIMVXLog::Instance().getLogger()->error(__VA_ARGS__)
// #define TIMVX_LOG_CRITICAL(...)    TIMVXLog::Instance().getLogger()->critical(__VA_ARGS__)

#define TIMVX_LOG_TRACE(...)       SPDLOG_TRACE(__VA_ARGS__)
#define TIMVX_LOG_DEBUG(...)       SPDLOG_DEBUG(__VA_ARGS__)
#define TIMVX_LOG_INFO(...)        SPDLOG_INFO(__VA_ARGS__)
#define TIMVX_LOG_WARN(...)        SPDLOG_WARN(__VA_ARGS__)
#define TIMVX_LOG_ERROR(...)       SPDLOG_ERROR(__VA_ARGS__)
#define TIMVX_LOG_CRITICAL(...)    SPDLOG_CRITICAL(__VA_ARGS__)


#define TIMVX_LOG_IMPL(level, ...)                                     \
do {                                                                   \
    switch(level)                                                      \
    {                                                                  \
        case TIMVX_LEVEL_TRACE:                                        \
            TIMVX_LOG_TRACE(__VA_ARGS__);                              \
            break;                                                     \
        case TIMVX_LEVEL_DEBUG:                                        \
            TIMVX_LOG_DEBUG(__VA_ARGS__);                              \
            break;                                                     \
        case TIMVX_LEVEL_INFO:                                         \
            TIMVX_LOG_INFO(__VA_ARGS__);                               \
            break;                                                     \
        case TIMVX_LEVEL_WARN:                                         \
            TIMVX_LOG_WARN(__VA_ARGS__);                               \
            break;                                                     \
        case TIMVX_LEVEL_ERROR:                                        \
            TIMVX_LOG_ERROR(__VA_ARGS__);                              \
            break;                                                     \
        case TIMVX_LEVEL_FATAL:                                        \
            TIMVX_LOG_CRITICAL(__VA_ARGS__);                           \
            break;                                                     \
        case TIMVX_LEVEL_OFF:                                          \
            break;                                                     \
        default:                                                       \
            std::string err_str = "unspported log level ...";          \
            TIMVX_LOG_CRITICAL(err_str, __VA_ARGS__);                  \
    }                                                                  \
} while(0)

#define TIMVX_LOG_FILE_NUM 3              // log file number: 3 
#define TIMVX_LOG_FILE_SIZE 10*1024*1024  // log file size: 10MB
#define TIMVX_LOG(level, ...)  TIMVX_LOG_IMPL(level, ##__VA_ARGS__)

namespace TimVX
{

    class TimVXLog : public NonCopyable
    {
    public:
        static TimVXLog& Instance();

        void initTimVXLog(std::string logger_name, std::string file_name, int log_level = spdlog::level::trace);

        void stopTimVXLog();

        void setLevel(int level = spdlog::level::trace);

        spdlog::logger* getLogger()
        {
            return m_logger.get();
        }

    private:
        TimVXLog() = default;
        ~TimVXLog();

    private:
        std::shared_ptr<spdlog::logger>              m_logger;
    };

}  // namespace TimVX