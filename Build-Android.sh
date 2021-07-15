
compiler=arm-linux-androideabi-clang++
target=armv7a-linux-androideabi$1-clang++
Compiler_dir=../android-ndk-r21e-linux-x86_64/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/

cp $Compiler_dir/$target $Compiler_dir/$compiler

XX=clang++ CC=clang scons Werror=0 -j16 debug=0 asserts=0 neon=1 opencl=1 os=android arch=armv7a 

rm $Compiler_dir/$compiler
