/***********************************
******  model_verify.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#include "tool_utils.h"
#include "timvx_model.h"

using namespace TimVX;

int parseModelVerifyOption(int argc, char* argv[], CmdLineArgOption& arg_opt)
{
    // 1 init arg options
    cxxopts::Options arg_options("model_verify", "model verify test");
    arg_options.add_options()
        // model weight file path
        ("weight", "Model weight file path", cxxopts::value<std::string>())
        // model para file path
        ("para", "Model para file path", cxxopts::value<std::string>())
        // log level, default is info level
        ("log_level", "log level", cxxopts::value<int>()->default_value("2"))
        // help
        ("help", "Print usage");
    arg_options.allow_unrecognised_options();

    // 2 parse arg
    auto parse_result = arg_options.parse(argc, argv);

    // 3 check help arg
    arg_opt.help_flag = false;
    if (parse_result.count("help"))
    {
        arg_opt.help_flag = true;
        std::cout << arg_options.help() << std::endl;
        return -1;
    }

    // 4 check unmatch arg
    const std::vector<std::string>& unmatch = parse_result.unmatched();
    if (parse_result.unmatched().size() > 0)
    {
        std::cout << "contain unsupported options:" << std::endl;
        for (int i = 0; i < unmatch.size(); i++)
            std::cout << unmatch[i] << std::endl;
        return -1;
    }

    // 5 chcek model/para file arg
    arg_opt.weight_file = "";
    if (0 == parse_result.count("weight"))
    {
        std::cout << "model weight file should be set" << std::endl;
        std::cout << arg_options.help() << std::endl;
        return -1;
    }
    arg_opt.weight_file = parse_result["weight"].as<std::string>();

    arg_opt.para_file = "";
    if (0 == parse_result.count("para"))
    {
        std::cout << "model para file should be set" << std::endl;
        std::cout << arg_options.help() << std::endl;
        return -1;
    }
    arg_opt.para_file = parse_result["para"].as<std::string>();

    // 6 check log arg
    // LOG_LEVEL_DEBUG = 1,
    // LOG_LEVEL_INFO,
    // LOG_LEVEL_WARN,
    // LOG_LEVEL_ERROR
    arg_opt.log_level = 1;
    arg_opt.log_level = parse_result["log_level"].as<int>();

    return 0;
}

int main(int argc, char* argv[])
{
    CmdLineArgOption cmd_option;
    if (0 != parseModelVerifyOption(argc, argv, cmd_option))
        return -1;

    std::shared_ptr<TimVXModel> model(new TimVXModel(cmd_option));
    if (nullptr == model.get())
        return -1;
    
    return 0;
}
