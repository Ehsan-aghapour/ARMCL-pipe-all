compiler=arm-linux-androideabi-clang++
#compiler=aarch64-linux-android-clang++

target=armv7a-linux-androideabi$1-clang++
#target=aarch64-linux-android$1-clang++

p=../android-ndk-r21e-linux-x86_64/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/
cp $p/$target $p/$compiler

XX=clang++ CC=clang scons Werror=0 -j16 debug=0 asserts=0 neon=1 opencl=1 os=android arch=armv7a 
#XX=clang++ CC=clang scons Werror=0 -j8 debug=0 asserts=0 neon=1 opencl=1 os=android arch=arm64-v8a

rm $p/$compiler
