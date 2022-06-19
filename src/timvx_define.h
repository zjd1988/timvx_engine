/***********************************
******  timvx_define.h
******
******  Created by zhaojd on 2022/06/12.
***********************************/

#pragma once

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define TIMVX_BUILD_FOR_IOS
#endif
#endif

#ifdef TIMVX_USE_LOGCAT
#include <android/log.h>
#define TIMVX_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MNNJNI", format, ##__VA_ARGS__)
#define TIMVX_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#elif defined TIMVX_BUILD_FOR_IOS
// on iOS, stderr prints to XCode debug area and syslog prints Console. You need both.
#include <syslog.h>
#define TIMVX_PRINT(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#define TIMVX_ERROR(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#else
#define TIMVX_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define TIMVX_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define TIMVX_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            TIMVX_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define TIMVX_ASSERT(x)
#endif

#define FUNC_PRINT(x) TIMVX_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) TIMVX_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define TIMVX_CHECK(success, log) \
if(!(success)){ \
TIMVX_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}

#if defined(_MSC_VER)
#if defined(BUILDING_TIMVX_DLL)
#define TIMVX_PUBLIC __declspec(dllexport)
#elif defined(USING_TIMVX_DLL)
#define TIMVX_PUBLIC __declspec(dllimport)
#else
#define TIMVX_PUBLIC
#endif
#else
#define TIMVX_PUBLIC __attribute__((visibility("default")))
#endif
