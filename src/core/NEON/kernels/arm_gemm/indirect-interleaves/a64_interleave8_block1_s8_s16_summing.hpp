/*
 * Copyright (c) 2019-2020 Arm Limited.
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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifdef __aarch64__

template<>
void interleave_block<8, 1, VLType::None, true>(
  int16_t * &out_ptr, const int8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first
)
{
  __asm__ __volatile__(
      "movi v1.8h, #0x0\n"
      "ldr x27, [%x[in], #0x0]\n"
      "mov x19, #0x0\n"
      "movi v0.4s, #0x0\n"
      "ldr x26, [%x[in], #0x8]\n"
      "cmp %x[height], #0x8\n"
      "movi v31.4s, #0x0\n"
      "ldr x25, [%x[in], #0x10]\n"
      "add x27, x27, %x[row_offset]\n"
      "ldr x24, [%x[in], #0x18]\n"
      "ldr x23, [%x[in], #0x20]\n"
      "add x26, x26, %x[row_offset]\n"
      "ldr x22, [%x[in], #0x28]\n"
      "add x25, x25, %x[row_offset]\n"
      "ldr x21, [%x[in], #0x30]\n"
      "add x24, x24, %x[row_offset]\n"
      "ldr x20, [%x[in], #0x38]\n"
      "add x23, x23, %x[row_offset]\n"
      "add x22, x22, %x[row_offset]\n"
      "add x21, x21, %x[row_offset]\n"
      "add x20, x20, %x[row_offset]\n"
      "beq 1f\n"
      "mov x20, x27\n"
      "cmp %x[height], #0x2\n"
      "csel x26, x26, x27, GE\n"
      "csel x25, x25, x27, GT\n"
      "cmp %x[height], #0x4\n"
      "csel x24, x24, x27, GE\n"
      "csel x23, x23, x27, GT\n"
      "cmp %x[height], #0x6\n"
      "csel x22, x22, x27, GE\n"
      "csel x21, x21, x27, GT\n"
      "1:"  // no_pointer_adj
      "prfm pldl1keep, [x27, #0x0]\n"
      "prfm pldl1keep, [x26, #0x0]\n"
      "prfm pldl1keep, [x25, #0x0]\n"
      "prfm pldl1keep, [x24, #0x0]\n"
      "prfm pldl1keep, [x23, #0x0]\n"
      "prfm pldl1keep, [x22, #0x0]\n"
      "prfm pldl1keep, [x21, #0x0]\n"
      "prfm pldl1keep, [x20, #0x0]\n"
      "prfm pldl1keep, [x27, #0x40]\n"
      "prfm pldl1keep, [x26, #0x40]\n"
      "prfm pldl1keep, [x25, #0x40]\n"
      "prfm pldl1keep, [x24, #0x40]\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "prfm pldl1keep, [x22, #0x40]\n"
      "prfm pldl1keep, [x21, #0x40]\n"
      "prfm pldl1keep, [x20, #0x40]\n"
      "cbnz %w[first], 2f\n"
      "sub %x[out_ptr], %x[out_ptr], #0x20\n"
      "ld1 { v0.4s }, [%x[out_ptr]]\n"
      "ldr q31, [%x[out_ptr], #0x10]\n"
      "2:"  // first_pass
      "cmp %x[width], #0x8\n"
      "blt 5f\n"
      "3:"  // Main loop head
      "cmp x19, #0xe\n"
      "ble 4f\n"
      "saddw v0.4s, v0.4s, v1.4h\n"
      "saddw2 v31.4s, v31.4s, v1.8h\n"
      "mov x19, #0x0\n"
      "movi v1.8h, #0x0\n"
      "4:"  // no_accumulate_16
      "ldr d30, [x27], #0x8\n"
      "prfm pldl1keep, [x27, #0x70]\n"
      "ldr d29, [x26], #0x8\n"
      "ldr d28, [x25], #0x8\n"
      "prfm pldl1keep, [x26, #0x70]\n"
      "ldr d27, [x24], #0x8\n"
      "prfm pldl1keep, [x25, #0x70]\n"
      "ldr d23, [x23], #0x8\n"
      "ldr d21, [x22], #0x8\n"
      "prfm pldl1keep, [x24, #0x70]\n"
      "ldr d26, [x21], #0x8\n"
      "ldr d25, [x20], #0x8\n"
      "prfm pldl1keep, [x23, #0x70]\n"
      "prfm pldl1keep, [x22, #0x70]\n"
      "sshll v30.8h, v30.8b, #0x0\n"
      "sshll v29.8h, v29.8b, #0x0\n"
      "prfm pldl1keep, [x21, #0x70]\n"
      "sshll v28.8h, v28.8b, #0x0\n"
      "prfm pldl1keep, [x20, #0x70]\n"
      "sshll v27.8h, v27.8b, #0x0\n"
      "sshll v23.8h, v23.8b, #0x0\n"
      "zip1 v24.8h, v30.8h, v23.8h\n"
      "sshll v21.8h, v21.8b, #0x0\n"
      "zip2 v23.8h, v30.8h, v23.8h\n"
      "sshll v26.8h, v26.8b, #0x0\n"
      "sshll v25.8h, v25.8b, #0x0\n"
      "zip1 v22.8h, v29.8h, v21.8h\n"
      "add x19, x19, #0x1\n"
      "zip2 v21.8h, v29.8h, v21.8h\n"
      "subs %x[width], %x[width], #0x8\n"
      "zip1 v20.8h, v28.8h, v26.8h\n"
      "cmp %x[width], #0x8\n"
      "zip1 v18.8h, v24.8h, v20.8h\n"
      "zip1 v19.8h, v27.8h, v25.8h\n"
      "zip1 v17.8h, v22.8h, v19.8h\n"
      "zip1 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x0]\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "zip2 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x10]\n"
      "zip2 v18.8h, v24.8h, v20.8h\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "zip2 v17.8h, v22.8h, v19.8h\n"
      "zip1 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x20]\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "zip2 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "zip2 v20.8h, v28.8h, v26.8h\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "zip1 v18.8h, v23.8h, v20.8h\n"
      "zip2 v19.8h, v27.8h, v25.8h\n"
      "zip1 v17.8h, v21.8h, v19.8h\n"
      "zip1 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x40]\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "zip2 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x50]\n"
      "zip2 v18.8h, v23.8h, v20.8h\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "zip2 v17.8h, v21.8h, v19.8h\n"
      "zip1 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x60]\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "zip2 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x70]\n"
      "add %x[out_ptr], %x[out_ptr], #0x80\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "bge 3b\n"
      "5:"  // Main loop skip
      "cbz %x[width], 10f\n"
      "tbz %x[width], #2, 7f\n"
      "ldr s30, [x27], #0x4\n"
      "ldr s29, [x26], #0x4\n"
      "ldr s28, [x25], #0x4\n"
      "ldr s27, [x24], #0x4\n"
      "ldr s23, [x23], #0x4\n"
      "ldr s21, [x22], #0x4\n"
      "ldr s26, [x21], #0x4\n"
      "ldr s25, [x20], #0x4\n"
      "tbz %x[width], #1, 6f\n"
      "ld1 { v30.h }[2], [x27], #0x2\n"
      "ld1 { v29.h }[2], [x26], #0x2\n"
      "ld1 { v28.h }[2], [x25], #0x2\n"
      "ld1 { v27.h }[2], [x24], #0x2\n"
      "ld1 { v23.h }[2], [x23], #0x2\n"
      "ld1 { v21.h }[2], [x22], #0x2\n"
      "ld1 { v26.h }[2], [x21], #0x2\n"
      "ld1 { v25.h }[2], [x20], #0x2\n"
      "mov x19, #0x6\n"
      "tbz %x[width], #0, 9f\n"
      "ld1 { v30.b }[6], [x27]\n"
      "ld1 { v29.b }[6], [x26]\n"
      "ld1 { v28.b }[6], [x25]\n"
      "ld1 { v27.b }[6], [x24]\n"
      "ld1 { v23.b }[6], [x23]\n"
      "ld1 { v21.b }[6], [x22]\n"
      "ld1 { v26.b }[6], [x21]\n"
      "ld1 { v25.b }[6], [x20]\n"
      "mov x19, #0x7\n"
      "b 9f\n"
      "6:"  // odd_loads_1_4
      "mov x19, #0x4\n"
      "tbz %x[width], #0, 9f\n"
      "ld1 { v30.b }[4], [x27]\n"
      "ld1 { v29.b }[4], [x26]\n"
      "ld1 { v28.b }[4], [x25]\n"
      "ld1 { v27.b }[4], [x24]\n"
      "ld1 { v23.b }[4], [x23]\n"
      "ld1 { v21.b }[4], [x22]\n"
      "ld1 { v26.b }[4], [x21]\n"
      "ld1 { v25.b }[4], [x20]\n"
      "mov x19, #0x5\n"
      "b 9f\n"
      "7:"  // odd_loads_2_0
      "tbz %x[width], #1, 8f\n"
      "ldr h30, [x27], #0x2\n"
      "ldr h29, [x26], #0x2\n"
      "ldr h28, [x25], #0x2\n"
      "ldr h27, [x24], #0x2\n"
      "ldr h23, [x23], #0x2\n"
      "ldr h21, [x22], #0x2\n"
      "ldr h26, [x21], #0x2\n"
      "ldr h25, [x20], #0x2\n"
      "mov x19, #0x2\n"
      "tbz %x[width], #0, 9f\n"
      "ld1 { v30.b }[2], [x27]\n"
      "ld1 { v29.b }[2], [x26]\n"
      "ld1 { v28.b }[2], [x25]\n"
      "ld1 { v27.b }[2], [x24]\n"
      "ld1 { v23.b }[2], [x23]\n"
      "ld1 { v21.b }[2], [x22]\n"
      "ld1 { v26.b }[2], [x21]\n"
      "ld1 { v25.b }[2], [x20]\n"
      "mov x19, #0x3\n"
      "b 9f\n"
      "8:"  // odd_loads_1_0
      "ldr b30, [x27, #0x0]\n"
      "ldr b29, [x26, #0x0]\n"
      "ldr b28, [x25, #0x0]\n"
      "ldr b27, [x24, #0x0]\n"
      "ldr b23, [x23, #0x0]\n"
      "ldr b21, [x22, #0x0]\n"
      "ldr b26, [x21, #0x0]\n"
      "ldr b25, [x20, #0x0]\n"
      "mov x19, #0x1\n"
      "9:"  // Odd load end
      "sshll v30.8h, v30.8b, #0x0\n"
      "sshll v29.8h, v29.8b, #0x0\n"
      "sshll v28.8h, v28.8b, #0x0\n"
      "sshll v27.8h, v27.8b, #0x0\n"
      "sshll v23.8h, v23.8b, #0x0\n"
      "zip1 v24.8h, v30.8h, v23.8h\n"
      "sshll v21.8h, v21.8b, #0x0\n"
      "sshll v26.8h, v26.8b, #0x0\n"
      "zip1 v20.8h, v28.8h, v26.8h\n"
      "sshll v25.8h, v25.8b, #0x0\n"
      "zip1 v22.8h, v29.8h, v21.8h\n"
      "subs x19, x19, #0x1\n"
      "zip1 v18.8h, v24.8h, v20.8h\n"
      "zip1 v19.8h, v27.8h, v25.8h\n"
      "zip1 v17.8h, v22.8h, v19.8h\n"
      "zip1 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x0]\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "add %x[out_ptr], %x[out_ptr], #0x10\n"
      "beq 10f\n"
      "zip2 v16.8h, v18.8h, v17.8h\n"
      "subs x19, x19, #0x1\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "str q16, [%x[out_ptr], #0x0]\n"
      "add %x[out_ptr], %x[out_ptr], #0x10\n"
      "beq 10f\n"
      "zip2 v18.8h, v24.8h, v20.8h\n"
      "zip2 v17.8h, v22.8h, v19.8h\n"
      "subs x19, x19, #0x1\n"
      "zip1 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x0]\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "add %x[out_ptr], %x[out_ptr], #0x10\n"
      "beq 10f\n"
      "zip2 v16.8h, v18.8h, v17.8h\n"
      "subs x19, x19, #0x1\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "str q16, [%x[out_ptr], #0x0]\n"
      "add %x[out_ptr], %x[out_ptr], #0x10\n"
      "beq 10f\n"
      "zip2 v23.8h, v30.8h, v23.8h\n"
      "zip2 v20.8h, v28.8h, v26.8h\n"
      "subs x19, x19, #0x1\n"
      "zip1 v18.8h, v23.8h, v20.8h\n"
      "zip2 v21.8h, v29.8h, v21.8h\n"
      "zip2 v19.8h, v27.8h, v25.8h\n"
      "zip1 v17.8h, v21.8h, v19.8h\n"
      "zip1 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x0]\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "add %x[out_ptr], %x[out_ptr], #0x10\n"
      "beq 10f\n"
      "zip2 v16.8h, v18.8h, v17.8h\n"
      "subs x19, x19, #0x1\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "str q16, [%x[out_ptr], #0x0]\n"
      "add %x[out_ptr], %x[out_ptr], #0x10\n"
      "beq 10f\n"
      "zip2 v18.8h, v23.8h, v20.8h\n"
      "zip2 v17.8h, v21.8h, v19.8h\n"
      "zip1 v16.8h, v18.8h, v17.8h\n"
      "str q16, [%x[out_ptr], #0x0]\n"
      "add v1.8h, v1.8h, v16.8h\n"
      "add %x[out_ptr], %x[out_ptr], #0x10\n"
      "10:"  // Odds skip
      "saddw v0.4s, v0.4s, v1.4h\n"
      "str q0, [%x[out_ptr], #0x0]\n"
      "saddw2 v31.4s, v31.4s, v1.8h\n"
      "str q31, [%x[out_ptr], #0x10]\n"
      "add %x[out_ptr], %x[out_ptr], #0x20\n"
      : [out_ptr] "+r" (out_ptr), [width] "+r" (width)
      : [first] "r" (first), [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset)
      : "cc", "memory", "v0", "v1", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27"
    );
}


#endif // __aarch64__
