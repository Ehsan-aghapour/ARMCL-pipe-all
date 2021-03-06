/*
 * Copyright (c) 2017-2020 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_LOGGING_MACROS_H
#define ARM_COMPUTE_LOGGING_MACROS_H

#include "arm_compute/core/utils/logging/LoggerRegistry.h"

#include <sstream>

#ifdef ARM_COMPUTE_LOGGING_ENABLED

#define ARM_COMPUTE_LOG_MSG(logger_name, log_level, msg)                                 \
    do                                                                                   \
    {                                                                                    \
        auto __logger = arm_compute::logging::LoggerRegistry::get().logger(logger_name); \
        if(__logger != nullptr)                                                          \
        {                                                                                \
            __logger->log(log_level, msg);                                               \
        }                                                                                \
    } while(false)

#define ARM_COMPUTE_LOG_MSG_WITH_FORMAT(logger_name, log_level, fmt, ...)                     \
    do                                                                                        \
    {                                                                                         \
        auto __logger = arm_compute::logging::LoggerRegistry::get().logger(logger_name);      \
        if(__logger != nullptr)                                                               \
        {                                                                                     \
            size_t size     = ::snprintf(nullptr, 0, fmt, __VA_ARGS__) + 1;                   \
            auto   char_str = std::make_unique<char[]>(size);                                 \
            ::snprintf(char_str.get(), size, #fmt, __VA_ARGS__);                              \
            __logger->log(log_level, std::string(char_str.get(), char_str.get() + size - 1)); \
        }                                                                                     \
    } while(false)

#define ARM_COMPUTE_LOG_STREAM(logger_name, log_level, stream)                           \
    do                                                                                   \
    {                                                                                    \
        auto __logger = arm_compute::logging::LoggerRegistry::get().logger(logger_name); \
        if(__logger != nullptr)                                                          \
        {                                                                                \
            std::ostringstream s;                                                        \
            s << stream;                                                                 \
            __logger->log(log_level, s.str());                                           \
        }                                                                                \
    } while(false)

#else /* ARM_COMPUTE_LOGGING_ENABLED */

#define ARM_COMPUTE_LOG_MSG(logger_name, log_level, msg)
#define ARM_COMPUTE_LOG_MSG_WITH_FORMAT(logger_name, log_level, fmt, ...)
#define ARM_COMPUTE_LOG_STREAM(logger_name, log_level, stream)

#endif /* ARM_COMPUTE_LOGGING_ENABLED */

#endif /* ARM_COMPUTE_LOGGING_MACROS_H */
