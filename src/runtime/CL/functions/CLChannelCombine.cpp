/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLChannelCombine.h"

#include "src/core/CL/kernels/CLChannelCombineKernel.h"

#include <utility>

using namespace arm_compute;

void CLChannelCombine::configure(const ICLTensor *plane0, const ICLTensor *plane1, const ICLTensor *plane2, const ICLTensor *plane3, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), plane0, plane1, plane2, plane3, output);
}

void CLChannelCombine::configure(const CLCompileContext &compile_context, const ICLTensor *plane0, const ICLTensor *plane1, const ICLTensor *plane2, const ICLTensor *plane3, ICLTensor *output)
{
    auto k = std::make_unique<CLChannelCombineKernel>();
    k->configure(compile_context, plane0, plane1, plane2, plane3, output);
    _kernel = std::move(k);
}

void CLChannelCombine::configure(const ICLImage *plane0, const ICLImage *plane1, const ICLImage *plane2, ICLMultiImage *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), plane0, plane1, plane2, output);
}

void CLChannelCombine::configure(const CLCompileContext &compile_context, const ICLImage *plane0, const ICLImage *plane1, const ICLImage *plane2, ICLMultiImage *output)
{
    auto k = std::make_unique<CLChannelCombineKernel>();
    k->configure(compile_context, plane0, plane1, plane2, output);
    _kernel = std::move(k);
}
