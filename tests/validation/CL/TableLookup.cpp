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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLTableLookup.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/CLLutAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/TableLookupFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(TableLookup)

template <typename T>
using CLTableLookupFixture = TableLookupValidationFixture<CLTensor, CLAccessor, CLTableLookup, CLLutAccessor<T>, CLLut, T>;
TEST_SUITE(U8)

FIXTURE_DATA_TEST_CASE(RunSmallU8, CLTableLookupFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLargeU8, CLTableLookupFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmallS16, CLTableLookupFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::S16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLargeS16, CLTableLookupFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::S16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
