/***********************************
******  model_compile.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#include "tool_utils.h"
#include "timvx_model.h"

using namespace TimVX;

int parseModelCompileOption(int argc, char* argv[], CmdLineArgOption& arg_opt)
{
    // 1 init arg options
    cxxopts::Options arg_options("model_compile", "model compile test");
    arg_options.add_options()
        // input model weight file path
        ("input_weight", "Input model weight file path", cxxopts::value<std::string>())
        // input model para file path
        ("input_para", "Input model para file path", cxxopts::value<std::string>())
        // output compiled model weight file path
        ("output_weight", "Output model weight file path", cxxopts::value<std::string>())
        // output compiled model para file path
        ("output_para", "Output model para file path", cxxopts::value<std::string>())
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

    // 5 check input model weight/para file arg
    arg_opt.weight_file = "";
    if (0 == parse_result.count("input_weight"))
    {
        std::cout << "input model file should be set" << std::endl;
        std::cout << arg_options.help() << std::endl;
        return -1;
    }
    arg_opt.weight_file = parse_result["input_weight"].as<std::string>();

    arg_opt.para_file = "";
    if (0 == parse_result.count("input_para"))
    {
        std::cout << "input model para file should be set" << std::endl;
        std::cout << arg_options.help() << std::endl;
        return -1;
    }
    arg_opt.para_file = parse_result["input_para"].as<std::string>();

    // 6 check output model weight/para file arg
    arg_opt.compile_weight_file = "";
    if (0 == parse_result.count("output_weight"))
    {
        std::cout << "output model weight file should be set" << std::endl;
        std::cout << arg_options.help() << std::endl;
        return -1;
    }
    arg_opt.compile_weight_file = parse_result["output_weight"].as<std::string>();

    arg_opt.compile_para_file = "";
    if (0 == parse_result.count("output_para"))
    {
        std::cout << "output model para file should be set" << std::endl;
        std::cout << arg_options.help() << std::endl;
        return -1;
    }
    arg_opt.compile_para_file = parse_result["output_para"].as<std::string>();

    // 7 check log arg
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
    if (0 != parseModelCompileOption(argc, argv, cmd_option))
        return -1;

    std::shared_ptr<TimVXModel> model(new TimVXModel(cmd_option));
    if (nullptr == model.get())
        return -1;
    
    return model->modelCompile();
}
