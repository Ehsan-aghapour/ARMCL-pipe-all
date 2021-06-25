
#compiler=arm-linux-androideabi-clang++
#target=armv7a-linux-androideabi$1-clang++
#p=../android-ndk-r21e-linux-x86_64/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/
#cp $p/$target $p/$compiler

scons Werror=0 -j16 debug=1 asserts=0 neon=1 opencl=1 os=linux arch=arm64-v8a

#rm $p/$compiler
