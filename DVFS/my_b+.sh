compiler=aarch64-linux-androideabi-clang++
#target=armv7a-linux-androideabi$1-clang++
target=aarch64-linux-android$1-clang++
p=/home/ehsan/UvA/ARMCL/android-ndk-r21e-linux-x86_64/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/
cp $p/$target $p/$compiler

#XX=clang++ CC=clang scons Werror=0 -j16 debug=0 asserts=0 neon=1 opencl=1 os=android arch=armv7a 

$compiler $2 -static-libstdc++ -lstdc++ -pie -Wl,-rpath,/system/usr/lib64/ -lz
#scons 

rm $p/$compiler

#wdir=/data/dataset/npu
#adb shell "mkdir -p /data/dataset/npu"
wdir=/system/
adb push a.out ${wdir}/
adb shell chmod +x ${wdir}/a.out
#adb shell ${wdir}/a.out model.rknn 20 
#adb shell ${wdir}/a.out mobilenet_v1_sample_test_precompiled.rknn ${wdir}/dog_224x224.jpg 20 
#adb shell ${wdir}/a.out model.rknn ${wdir}/space_shuttle_227.jpg 1

