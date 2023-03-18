
./b64.sh 23
##adb push build/examples/Pipeline/graph_yolov3_n_pipe_npu /data/data/com.termux/files/home/ARMCL-RockPi/test_graph/
##adb shell /data/data/com.termux/files/home/ARMCL-RockPi/test_graph/graph_yolov3_n_pipe_npu $1 $2 $3 $4 $5
adb push build/examples/NPU/graph_alexnet_n_pipe_npu /data/data/com.termux/files/home/ARMCL-RockPi/test_graph/
adb shell /data/data/com.termux/files/home/ARMCL-RockPi/test_graph/graph_alexnet_n_pipe_npu $1 $2 $3 $4 $5 $6 $7
