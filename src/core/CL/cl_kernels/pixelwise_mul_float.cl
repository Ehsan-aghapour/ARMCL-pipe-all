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
#include "helpers.h"

#ifdef SATURATE
#define CONVERT_OP_FLOAT_STR(x, type, round) (convert_##type##_sat##round(x))
#else /* SATURATE */
#define CONVERT_OP_FLOAT_STR(x, type, round) (convert_##type##round(x))
#endif /* SATURATE */
#define CONVERT_OP_FLOAT(x, type, round) CONVERT_OP_FLOAT_STR(x, type, round)

#if defined(DATA_TYPE_IN1) && defined(DATA_TYPE_IN2) && defined(ACC_DATA_TYPE) && defined(DATA_TYPE_OUT)

#if defined(ACTIVATION_TYPE)
#include "activation_float_helpers.h"
#endif // defined(ACTIVATION_TYPE)

/** Performs a pixelwise multiplication with float scale of either integer or float inputs.
 *
 * @attention The inputs and output data types need to be passed at compile time using -DDATA_TYPE_IN1, -DDATA_TYPE_IN2 and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN1=uchar -DDATA_TYPE_IN2=ushort -DDATA_TYPE_OUT=short
 * @attention The data type of the intermediate result of the multiplication should passed as well using -DACC_DATA_TYPE.
 * e.g. If one of inputs is S16 -DACC_DATA_TYPE=int should be passed else -DACC_DATA_TYPE=short.
 * @attention -DDATA_TYPE_FLOAT must be passed if floating point inputs are provided.
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: U8, S16, F16, F32
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_z                        in1_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: U8, S16, F16, F32
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_z                        in2_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8, S16, F16, F32
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  scale                             Float scaling factor. Supported data types: F32
 */
__kernel void pixelwise_mul_float(
    TENSOR3D_DECLARATION(in1),
    TENSOR3D_DECLARATION(in2),
    TENSOR3D_DECLARATION(out),
    const float scale)
{
    // Get pixels pointer
    Tensor3D in1 = CONVERT_TO_TENSOR3D_STRUCT(in1);
    Tensor3D in2 = CONVERT_TO_TENSOR3D_STRUCT(in2);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Load data
    VEC_DATA_TYPE(ACC_DATA_TYPE, 16)
    in1_data = CONVERT(vload16(0, (__global DATA_TYPE_IN1 *)in1.ptr), VEC_DATA_TYPE(ACC_DATA_TYPE, 16));
    VEC_DATA_TYPE(ACC_DATA_TYPE, 16)
    in2_data = CONVERT(vload16(0, (__global DATA_TYPE_IN2 *)in2.ptr), VEC_DATA_TYPE(ACC_DATA_TYPE, 16));

    // Perform multiplication
#ifdef DATA_TYPE_FLOAT
    VEC_DATA_TYPE(DATA_TYPE_OUT, 16)
    res = CONVERT(in1_data * in2_data * (ACC_DATA_TYPE)scale, VEC_DATA_TYPE(DATA_TYPE_OUT, 16));
#else  /* DATA_TYPE_FLOAT */
    VEC_DATA_TYPE(DATA_TYPE_OUT, 16)
    res = CONVERT_OP_FLOAT(CONVERT_OP_FLOAT((convert_float16(in1_data * in2_data) * scale), VEC_DATA_TYPE(ACC_DATA_TYPE, 16), ROUND), VEC_DATA_TYPE(DATA_TYPE_OUT, 16), ROUND);
#endif /* DATA_TYPE_FLOAT */

#if defined(ACTIVATION_TYPE)
    vstore16(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE_OUT, VEC_SIZE, res, A_VAL, B_VAL), 0, (__global DATA_TYPE_OUT *)out.ptr);
#else  // defined(ACTIVATION_TYPE)
    // Store result
    vstore16(res, 0, (__global DATA_TYPE_OUT *)out.ptr);
#endif // defined(ACTIVATION_TYPE)
}
#endif /* defined(DATA_TYPE_IN1) && defined(DATA_TYPE_IN2) && defined(ACC_DATA_TYPE) && defined(DATA_TYPE_OUT) */

#if defined(DATA_TYPE)

/** Performs a pixelwise multiplication of complex float values
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: F16/F32
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_z                        in1_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: same as @p in1_ptr
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_z                        in2_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: same as @p in1_ptr
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void pixelwise_mul_complex(
    TENSOR3D_DECLARATION(in1),
    TENSOR3D_DECLARATION(in2),
    TENSOR3D_DECLARATION(out))
{
    // Get pixels pointer
    Tensor3D in1 = CONVERT_TO_TENSOR3D_STRUCT(in1);
    Tensor3D in2 = CONVERT_TO_TENSOR3D_STRUCT(in2);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, 2)
    vin1 = vload2(0, (__global DATA_TYPE *)in1.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    vin2 = vload2(0, (__global DATA_TYPE *)in2.ptr);

    // Perform complex multiplication
    VEC_DATA_TYPE(DATA_TYPE, 2)
    res = { vin1.x *vin2.x - vin1.y * vin2.y, vin1.x *vin2.y + vin2.x * vin1.y };

#if defined(ACTIVATION_TYPE)
    vstore2(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, res, A_VAL, B_VAL), 0, (__global DATA_TYPE *)out.ptr);
#else  // defined(ACTIVATION_TYPE)
    // Store result
    vstore2(res, 0, (__global DATA_TYPE *)out.ptr);
#endif // defined(ACTIVATION_TYPE)
}

#endif // defined(DATA_TYPE)