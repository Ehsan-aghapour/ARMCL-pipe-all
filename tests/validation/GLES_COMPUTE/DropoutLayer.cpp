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

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCDropoutLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DropoutLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
//constexpr AbsoluteTolerance<float> tolerance_f(0.00001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */

const auto testparam = combine(framework::dataset::make("ratio", { 0.5f }), framework::dataset::make("forward", { false, true }));
} // namespace

TEST_SUITE(GC)
TEST_SUITE(DropoutLayer)

template <typename T>
using GCDropoutLayerFixture = DropoutLayerValidationFixture<GCTensor, GCAccessor, GCDropoutLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(Random, GCDropoutLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::SmallShapes(), testparam), framework::dataset::make("DataType", DataType::F32)))
{
    // FIXME(APPBROWSER-302)
    // Validate output
    //validate(GCAccessor(_target), _reference, tolerance_f, 0);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Random, GCDropoutLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::SmallShapes(), testparam), framework::dataset::make("DataType", DataType::F16)))
{
    // FIXME(APPBROWSER-302)
    // Validate output
    //validate(GCAccessor(_target), _reference, tolerance_f, 0);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
