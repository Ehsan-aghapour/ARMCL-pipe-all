compiler=aarch64-linux-androideabi-clang++
target=aarch64-linux-android$1-clang++
p=/home/ehsan/UvA/ARMCL/android-ndk-r21e-linux-x86_64/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/
cp $p/$target $p/$compiler

#XX=clang++ CC=clang scons Werror=0 -j16 debug=0 asserts=0 neon=1 opencl=1 os=android arch=armv7a 

#$compiler $2 -Iinclude/applib/ovxinc/include/ -Iinclude/service/ovx_inc/  -Llib/ -lovxlib -ljpeg_t -lvnn_inceptionv3 -Wl,-rpath,/system/usr/lib/
scons 

rm $p/$compiler
