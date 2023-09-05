/********************************************
// Filename: non_copyable.h
// Created by zhaojiadi on 2021/9/27
// Description:

********************************************/
#pragma once

namespace TimVX
{
    
    /** protocol class. used to delete assignment operator. */
    class NonCopyable
    {
    public:
        NonCopyable()                    = default;
        NonCopyable(const NonCopyable&)  = delete;
        NonCopyable(const NonCopyable&&) = delete;
        NonCopyable& operator=(const NonCopyable&) = delete;
        NonCopyable& operator=(const NonCopyable&&) = delete;
    };

} // namespace TimVX