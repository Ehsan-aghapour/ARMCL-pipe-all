# ARMCL-PipeALL
Pipe-line implementation in ARM Compute Library (See pipe-all branch).

git clone https://github.com/Ehsan-aghapour/ARMCL-PipeALL.git -b pipe-all

<br/>
<br/>


# Compiling for Android

First it is required to prepare cross compile tools to compile source code in linux system for android target. Here is the steps to download and settup tools.

1- Download Android NDK:
https://developer.android.com/ndk/downloads

2- We should create a standalone toolchains for compiling source code for android. Based on your platform set --arch to arm or arm64 in the following command. $corss-compile-dir is your arbitrary dir at which cross compile toolchains will be created.

$NDK/build/tools/make_standalone_toolchain.py --arch arm/arm64 --api 23 --stl gnustl --install-dir $cross_compile_dir

This command create cross compile toolchains at $cross-compile-dir.

3- Add $cross-compile-dir/bin to the path:
export PATH=$cross-compile-dir/bin/:$PATH

4- Go to the ARMCL source dir (cd $ARMCL-source-dir) and use the following command to compile it:
CXX=clang++ CC=clang scons Werror=0 debug=0 asserts=0 neon=1 opencl=1 os=android arch=arm64-v8a -j8


<div align="center">
 <img src="https://raw.githubusercontent.com/ARM-software/ComputeLibrary/gh-pages/ACL_logo.png"><br><br>
</div>

Release repository: https://github.com/arm-software/ComputeLibrary

Development repository: https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary

Please report issues here: https://github.com/ARM-software/ComputeLibrary/issues

**Make sure you are using the latest version of the library before opening an issue. Thanks**

News:

- [Gian Marco's talk on Performance Analysis for Optimizing Embedded Deep Learning Inference Software](https://www.embedded-vision.com/platinum-members/arm/embedded-vision-training/videos/pages/may-2019-embedded-vision-summit)
- [Gian Marco's talk on optimizing CNNs with Winograd algorithms at the EVS](https://www.embedded-vision.com/platinum-members/arm/embedded-vision-training/videos/pages/may-2018-embedded-vision-summit-iodice)
- [Gian Marco's talk on using SGEMM and FFTs to Accelerate Deep Learning](https://www.embedded-vision.com/platinum-members/arm/embedded-vision-training/videos/pages/may-2016-embedded-vision-summit-iodice)

Related projects:

- [Arm NN SDK](https://github.com/arm-software/armnn)

Tutorials:

- [Tutorial: Cartoonifying Images on Raspberry Pi with the Compute Library](https://community.arm.com/graphics/b/blog/posts/cartoonifying-images-on-raspberry-pi-with-the-compute-library)
- [Tutorial: Running AlexNet on Raspberry Pi with Compute Library](https://community.arm.com/processors/b/blog/posts/running-alexnet-on-raspberry-pi-with-compute-library)

Documentation (API, changelogs, build guide, contribution guide, errata, etc.) available at https://github.com/ARM-software/ComputeLibrary/wiki/Documentation.

Binaries available at https://github.com/ARM-software/ComputeLibrary/releases.

### Supported Architectures/Technologies

- Arm® CPUs:
    - Arm® Cortex®-A processor family using Arm® Neon™ technology
    - Arm® Cortex®-R processor family with Armv8-R AArch64 architecture using Arm® Neon™ technology
    - Arm® Cortex®-X1 processor using Arm® Neon™ technology

- Arm® Mali™ GPUs:
    - Arm® Mali™-G processor family
    - Arm® Mali™-T processor family

- x86

### Supported OS

- Android™
- Bare Metal
- Linux®
- macOS®
- Tizen™

## License and Contributions

The software is provided under MIT license. Contributions to this project are accepted under the same license.

### Public mailing list
For technical discussion, the ComputeLibrary project has a public mailing list: acl-dev@lists.linaro.org
The list is open to anyone inside or outside of Arm to self subscribe.  In order to subscribe, please visit the following website:
https://lists.linaro.org/mailman/listinfo/acl-dev

### Developer Certificate of Origin (DCO)
Before the ComputeLibrary project accepts your contribution, you need to certify its origin and give us your permission. To manage this process we use the Developer Certificate of Origin (DCO) V1.1 (https://developercertificate.org/)

To indicate that you agree to the the terms of the DCO, you "sign off" your contribution by adding a line with your name and e-mail address to every git commit message:

```Signed-off-by: John Doe <john.doe@example.org>```

You must use your real name, no pseudonyms or anonymous contributions are accepted.

## Trademarks and Copyrights

Android is a trademark of Google LLC.

Arm, Cortex and Mali are registered trademarks or trademarks of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries.

Mac and macOS are trademarks of Apple Inc., registered in the U.S. and other
countries.

Tizen is a registered trademark of The Linux Foundation.
