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
#include "arm_compute/runtime/CL/functions/CLHOGDescriptor.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLHOGDescriptorKernel.h"
#include "src/core/CL/kernels/CLMagnitudePhaseKernel.h"

using namespace arm_compute;

CLHOGDescriptor::CLHOGDescriptor(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _gradient(),
      _orient_bin(std::make_unique<CLHOGOrientationBinningKernel>()),
      _block_norm(std::make_unique<CLHOGBlockNormalizationKernel>()),
      _mag(),
      _phase(),
      _hog_space()
{
}

CLHOGDescriptor::~CLHOGDescriptor() = default;

void CLHOGDescriptor::configure(ICLTensor *input, ICLTensor *output, const IHOG *hog, BorderMode border_mode, uint8_t constant_border_value)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, hog, border_mode, constant_border_value);
}

void CLHOGDescriptor::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const IHOG *hog, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    ARM_COMPUTE_ERROR_ON(nullptr == hog);

    const HOGInfo *hog_info = hog->info();
    const size_t   width    = input->info()->dimension(Window::DimX);
    const size_t   height   = input->info()->dimension(Window::DimY);
    const size_t   num_bins = hog_info->num_bins();

    Size2D cell_size = hog_info->cell_size();

    // Calculate number of cells along the x and y directions for the hog_space
    const size_t num_cells_x = width / cell_size.width;
    const size_t num_cells_y = height / cell_size.height;

    // TensorShape of the input image
    const TensorShape &shape_img = input->info()->tensor_shape();

    // TensorShape of the hog space
    TensorShape shape_hog_space = input->info()->tensor_shape();
    shape_hog_space.set(Window::DimX, num_cells_x);
    shape_hog_space.set(Window::DimY, num_cells_y);

    // Intitialize tensors for magnitude, phase and hog space
    TensorInfo info_mag(shape_img, Format::S16);
    _mag.allocator()->init(info_mag);

    TensorInfo info_phase(shape_img, Format::U8);
    _phase.allocator()->init(info_phase);

    TensorInfo info_space(shape_hog_space, num_bins, DataType::F32);
    _hog_space.allocator()->init(info_space);

    // Manage intermediate buffers
    _memory_group.manage(&_mag);
    _memory_group.manage(&_phase);

    // Initialise gradient kernel
    _gradient.configure(compile_context, input, &_mag, &_phase, hog_info->phase_type(), border_mode, constant_border_value);

    // Manage intermediate buffers
    _memory_group.manage(&_hog_space);

    // Initialise orientation binning kernel
    _orient_bin->configure(compile_context, &_mag, &_phase, &_hog_space, hog->info());

    // Initialize HOG norm kernel
    _block_norm->configure(compile_context, &_hog_space, output, hog->info());

    // Allocate intermediate tensors
    _mag.allocator()->allocate();
    _phase.allocator()->allocate();
    _hog_space.allocator()->allocate();
}

void CLHOGDescriptor::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    // Run gradient
    _gradient.run();

    // Run orientation binning
    CLScheduler::get().enqueue(*_orient_bin, false);

    // Run block normalization
    CLScheduler::get().enqueue(*_block_norm);
}