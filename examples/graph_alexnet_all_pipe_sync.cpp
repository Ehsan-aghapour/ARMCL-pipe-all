/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef My_print
#include "arm_compute/gl_vs.h"
#endif

//#include <sys/wait.h>
//#include <sys/list.h>
//wait_queue_head_t wq;
//int flag = 0;



//Ehsan
#include<chrono>
#include<thread>
#include <sys/types.h>
#include <dirent.h>
//#include<unistd.h>
//#include<sched.h>
#include "arm_compute/graph/Types.h"
#include "arm_compute/gl_vs.h"


#include "arm_compute/graph.h"
#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/Utils.h"
#endif /* ARM_COMPUTE_CL */
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;



int core0=0;
int core1=0;
int core2=0;

double process_times[3]={0.0};

//Ehsan 
typedef std::vector<std::string> stringvec;
void read_directory(const std::string& name, stringvec& v)
{

    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        if(arm_compute::utility::endswith(dp->d_name, ".ppm"))
           v.push_back(name+(dp->d_name));
    }

    closedir(dirp);
}

//Ehsan
size_t image_index=0;
stringvec images_list;
bool imgs=0;
bool ann=0;

/** Example demonstrating how to implement AlexNet's network using the Compute Library's graph API */
class GraphAlexnetExample : public Example
{
public:
    GraphAlexnetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "AlexNet") , common_params2(), graph2(1,"Alexnet2"), common_params3(), graph3(2,"Alexnet3")
    {
    }

    void switch_graph(){
        // Finalize graph
        GraphConfig config;
        if(cluster==0){
        	config.num_threads = common_params.threads2;
        	//config.cluster=0;
        }
        else{
        	config.num_threads = common_params.threads;
        	//config.cluster=1;
        }
        //std::cout<<"n trds:"<<config.num_threads<<std::endl;
        config.cluster=cluster;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;

        //config.cluster=0;

        if(cluster>-1)
        	graph.finalize(common_params.target, config);

        //arm_compute::graph::Tensor *f_out;
        for(auto &node : graph.graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
            {
                f_out=node->input(0);
                 continue;
            }
        }

#if My_print > 0
        std::cout<<"Setup Second Graph,\n"<<f_out->desc().shape<<std::endl;
        //std::cout<<f_out->desc().shape.x()<<','<<f_out->desc().shape.y()<<','<<f_out->desc().shape.z()<<std::endl;
#endif

        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core1,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
        // Create input descriptor
        //const auto        operation_layout = common_params.data_layout;
        ////const TensorShape tensor_shape2     = permute_shape(TensorShape(13U, 13U, 384U, 1U), DataLayout::NCHW, operation_layout);
        const TensorShape tensor_shape2 = f_out->desc().shape;
        const auto        operation_layout = common_params.data_layout;
        input_descriptor2 = TensorDescriptor(tensor_shape2, common_params.data_type).set_layout(operation_layout);


    }

    void switch_graph2(){
        // Finalize graph
        GraphConfig config;

        if(cluster2==0){
        	//config.num_threads = common_params.threads2;
        	config.cluster = 0;
        }
        else{
        	//config.num_threads = common_params.threads;
        	config.cluster = 1;
        }
        config.cluster=cluster2;
        //config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;

        //config.cluster=0;

        if(cluster2>-1)
        	graph2.finalize(common_params2.target, config);

        //arm_compute::graph::Tensor *f_out;
        for(auto &node : graph2.graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
            {
                s_out=node->input(0);
                continue;
            }
        }

        for(auto &node : graph2.graph().nodes())
		{
			if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
			{
				s_in=node->output(0);
				continue;
			}
		}

#if My_print > 0
        std::cout<<"Setup Third Graph,\n"<<s_out->desc().shape<<std::endl;
        //std::cout<<f_out->desc().shape.x()<<','<<f_out->desc().shape.y()<<','<<f_out->desc().shape.z()<<std::endl;
#endif

        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core2,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
        // Create input descriptor
        //const auto        operation_layout = common_params.data_layout;
        ////const TensorShape tensor_shape2     = permute_shape(TensorShape(13U, 13U, 384U, 1U), DataLayout::NCHW, operation_layout);
        const TensorShape tensor_shape3 = s_out->desc().shape;
        const auto        operation_layout = common_params2.data_layout;
        input_descriptor3 = TensorDescriptor(tensor_shape3, common_params2.data_type).set_layout(operation_layout);


    }




    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);
        common_params2 = consume_common_graph_parameters(common_opts);
        
	    //Ehsan
	    imgs=!(common_params.image.empty());
	    if(imgs){
	   read_directory(common_params.image,images_list);
	   std::cout<<images_list.size()<<" Input images are read from "<<common_params.image<<std::endl;
	   common_params.image=images_list[image_index];
        }

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");


        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<CaffePreproccessor>(mean_rgb);

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape     = permute_shape(TensorShape(227U, 227U, 3U, 1U), DataLayout::NCHW, operation_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;




        //Ehsan
               //**********************************************************************************
               common_params2 = consume_common_graph_parameters(common_opts);

               common_params3 = consume_common_graph_parameters(common_opts);

               //common_params.target=arm_compute::graph::Target ::CL;
               //common_params2.target=arm_compute::graph::Target ::NEON;
               //std::cout<<int(common_params.target)<<std::endl<<std::endl;

               ////common_params2.target=static_cast<arm_compute::graph::Target>(3-int(common_params.target));
               std::string order=common_params.order;
       		if(order[0]=='B'){
       			common_params.target=static_cast<arm_compute::graph::Target>(1);
       			cluster=1;
       		}
       		if(order[0]=='L'){
       			common_params.target=static_cast<arm_compute::graph::Target>(1);
       			cluster=0;
       		}
       		if(order[0]=='G'){
       			common_params.target=static_cast<arm_compute::graph::Target>(2);
       			cluster=2;
       			gpu_index=0;
       		}
       		if (order[0]=='*'){
       			//common_params.target=static_cast<arm_compute::graph::Target>(1);
       			cluster=-1;
       		}

       		if(order[2]=='B'){
       			common_params2.target=static_cast<arm_compute::graph::Target>(1);
       			cluster2=1;
       		}
       		if(order[2]=='L'){
       			common_params2.target=static_cast<arm_compute::graph::Target>(1);
       			cluster2=0;
       		}
       		if(order[2]=='G'){
       			common_params2.target=static_cast<arm_compute::graph::Target>(2);
       			cluster2=2;
       			gpu_index=1;
       		}
       		if(order[2]=='*'){
       			cluster2=-1;
       		}

       		if(order[4]=='B'){
       			common_params3.target=static_cast<arm_compute::graph::Target>(1);
       			cluster3=1;
       		}
       		if(order[4]=='L'){
       			common_params3.target=static_cast<arm_compute::graph::Target>(1);
       			cluster3=0;
       		}
       		if(order[4]=='G'){
       			common_params3.target=static_cast<arm_compute::graph::Target>(2);
       			cluster3=2;
       			gpu_index=2;
       		}
       		if(order[4]=='*'){
       			cluster3=-1;
       		}

       		std::map<int, int> core = {{0, 1}, {1, 5}, {2, 4},{-1,3}};
       		core0=core[cluster];
       		core1=core[cluster2];
       		core2=core[cluster3];

       		//std::cout<<"\nFirst graph manages on core:"<<core0<<std::endl;
       		cpu_set_t set;
       		CPU_ZERO(&set);
       		CPU_SET(core0,&set);
       		ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");


            common_params.labels="transfer_wait";
            if(order[0]=='G'){
            	common_params.labels="transfer";
            }
            if(order[0]=='*' || order[2]=='*'){
            	common_params.labels="";
            }
            common_params2.labels="transfer2_wait";
       		if(order[2]=='G'){
       			common_params2.labels="transfer2";
       		}
       		if (order[2]=='*' || order[4]=='*'){
       			common_params2.labels="";
       		}



		   common_params2.image="transfer";
		   if(order[0]=='G'){
			common_params2.image="transfer_wait";
		   }
		   if(order[0]=='*' || order[2]=='*'){
			   common_params2.image="";
		   }
		   common_params3.image="transfer2";
		   if(order[2]=='G'){
			common_params3.image="transfer2_wait";
		   }
		   if (order[2]=='*' || order[4]=='*'){
			   common_params3.image="";
		   }
		   //common_params.threads=4;

		   p=common_params.partition_point;
		   p2=common_params.partition_point2;

		   // Print parameter values
		   //std::cout << common_params << std::endl;
		   // Print parameter values
		   // Print parameter values
		   std::cout << common_params << std::endl;

		   /*std::cout <<"\nGraph2:\n"<< common_params2 << std::endl;

		   std::cout <<"\nGraph3:\n"<< common_params3 << std::endl;*/


        Stream *sub_graph=&graph;
        CommonGraphParams *_common_params=&common_params;
        int Layer=0;
        //bool second=false;
        annotate=common_params.annotate;
        ann=annotate;
        //save_model=common_params.save;


        (*sub_graph) << _common_params->target
              << _common_params->fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(*_common_params, std::move(preprocessor)))
              // Layer 1
              << ConvolutionLayer(
                  11U, 11U, 96U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
                  PadStrideInfo(4, 4, 0, 0))
              .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool1");

        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;

        }
        if(Layer==p2){
        	//common_params2.labels="transfer2";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph3;
        	_common_params=&common_params3;
        	switch_graph2();
        	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
        	third=true;

        }




              // Layer 2
        (*sub_graph) << ConvolutionLayer(
                  5U, 5U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_b.npy"),
                  PadStrideInfo(1, 1, 2, 2), 2)
              .set_name("conv2")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm2")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool2");


        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;

        }
        if(Layer==p2){
        	//common_params2.labels="transfer2";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph3;
        	_common_params=&common_params3;
        	switch_graph2();
        	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
        	third=true;

        }


        // Layer 3
        (*sub_graph)<< ConvolutionLayer(
		                  3U, 3U, 384U,
		                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy", weights_layout),
		                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"),
		                  PadStrideInfo(1, 1, 1, 1))
		              .set_name("conv3")
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3");


        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;

        }
        if(Layer==p2){
        	//common_params2.labels="transfer2";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph3;
        	_common_params=&common_params3;
        	switch_graph2();
        	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
        	third=true;

        }


        // Layer 4
        (*sub_graph)<< ConvolutionLayer(
            3U, 3U, 384U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"),
            PadStrideInfo(1, 1, 1, 1), 2)
        .set_name("conv4")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4");


        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;

        }
        if(Layer==p2){
        	//common_params2.labels="transfer2";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph3;
        	_common_params=&common_params3;
        	switch_graph2();
        	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
        	third=true;

        }


        // Layer 5
        (*sub_graph)<< ConvolutionLayer(
            3U, 3U, 256U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"),
            PadStrideInfo(1, 1, 1, 1), 2)
        .set_name("conv5")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5")
        << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool5");



        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;

        }
        if(Layer==p2){
        	//common_params2.labels="transfer2";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph3;
        	_common_params=&common_params3;
        	switch_graph2();
        	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
        	third=true;

        }

        // Layer 6
        (*sub_graph)<< FullyConnectedLayer(
            4096U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy"))
        .set_name("fc6")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu6");


        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;

        }
        if(Layer==p2){
        	//common_params2.labels="transfer2";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph3;
        	_common_params=&common_params3;
        	switch_graph2();
        	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
        	third=true;

        }

        // Layer 7
        (*sub_graph)<< FullyConnectedLayer(
            4096U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy"))
        .set_name("fc7")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu7");

        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;

        }
        if(Layer==p2){
        	//common_params2.labels="transfer2";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph3;
        	_common_params=&common_params3;
        	switch_graph2();
        	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
        	third=true;

        }

        if(!second){
        	common_params.labels=common_params3.labels;
        }
        else if(!third){
			common_params2.labels=common_params3.labels;
		}

        // Layer 8
        (*sub_graph)<< FullyConnectedLayer(
            1000U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy"))
        .set_name("fc8")
        // Softmax
        << SoftmaxLayer().set_name("prob")
        << OutputLayer(get_output_accessor(*_common_params, 5));


        // Finalize graph
        GraphConfig config;
        int total_graphs=second+third+1;
        if(total_graphs==3)
        	config.cluster = cluster3;
        if(total_graphs==2)
            config.cluster = cluster2;
        if(total_graphs==1)
            config.cluster = cluster;

        if(config.cluster==0){
        	config.num_threads = common_params.threads2;
        }
        else if(config.cluster==1){
        	config.num_threads = common_params.threads;
        }
        //config.num_threads = _common_params->threads;
        config.use_tuner   = _common_params->enable_tuner;
        config.tuner_mode  = _common_params->tuner_mode;
        config.tuner_file  = _common_params->tuner_file;
        config.mlgo_file   = _common_params->mlgo_file;
        //config.cluster=0;

        if (config.cluster > -1)
        	sub_graph->finalize(_common_params->target, config);


        //arm_compute::graph::Tensor *s_in;
        for(auto &node : sub_graph->graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
            {
            	if(third && cluster3>-1)
            		t_in = node->output(0);
            	else if(second && cluster2>-1)
            		s_in = node->output(0);
            }
        }

        std::cout<<"First partition point:"<<p<<std::endl;
        std::cout<<"Second partition point:"<<p2<<std::endl;
        std::cout<<"Total parts:"<<Layer+1<<std::endl<<std::endl;


        // Save the opencl kernels to a file
        if(common_opts.enable_cl_cache)
        {
#ifdef ARM_COMPUTE_CL
            save_program_cache_to_file();
#endif /* ARM_COMPUTE_CL */
        }

        return true;
    }

    void do_run() override
    {
        // Run graph
        //Ehsan
    	std::thread First(&GraphAlexnetExample::do_run_1,this,core0);
    	std::thread Second(&GraphAlexnetExample::do_run_2,this,core1);
    	std::thread Third(&GraphAlexnetExample::do_run_3,this,core2);
    	First.join();
    	Second.join();
    	Third.join();
    	double max=process_times[0];
    	if (process_times[1]>max){
    		max=process_times[1];
    	}
    	if (process_times[2]>max){
    		max=process_times[2];
    	}
    	double throughput=1000.0/max;
    	double latency=process_times[0]+process_times[1]+process_times[2];
    	std::cout<<"\n************************************************\n\n";
    	std::cout<<"Frame rate is: "<<throughput<<" FPS"<<std::endl;
    	std::cout<<"Frame latency is: "<<latency<<" ms"<<std::endl;
    	std::cout<<"\n************************************************\n";
    }


    void do_run_1(int core_id)
    {
        // Run graph
        //Ehsan
    	//int core_id=1;
/*
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core_id, &set);
        CPU_SET(5,&set);
        CPU_SET(4,&set);
        CPU_SET(3,&set);
        CPU_SET(2,&set);
        CPU_SET(1,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
*/

    	if(cluster==-1)
    		return;
        cpu_set_t set;
        CPU_ZERO(&set);
        //CPU_SET(core_id, &set);
        //CPU_SET(1,&set);
        CPU_SET(core0,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");


        //std::cerr<<"start running first graph ...\n";
        ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
        double in=0;
        double task=0.0001;
        double out=0;
        int tt=(common_params.n);
        auto tstart=std::chrono::high_resolution_clock::now();
        //std::cout<<tstart.time_since_epoch().count()<<std::endl;
        //std::cout<<tt<<std::endl;
        if(p==111){
			std::cout<<"Press key to start running...\n";
			std::string g;
			std::cin>>g;
        }
        for(int i=0;i<(tt+1);i++){
        	if(i==1){
        		std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        		std::cerr<<"\nRunning Inference ... ";
        		tstart=std::chrono::high_resolution_clock::now();
        		//std::cout<<tstart.time_since_epoch().count()<<std::endl;
        		in=task=out=0;
        		task=0;
        	}
			if(imgs){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				std::cout<<"\n\nFirst graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				im_acc->set_filename(images_list[image_index++]);
			}

            graph.run(in,task,out,annotate);
        }
        auto tfinish=std::chrono::high_resolution_clock::now();
        //std::cout<<tfinish.time_since_epoch().count()<<std::endl;
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=1000*cost0/tt;
        in=1000*in/tt;
        task=1000*task/tt;
        out=1000*out/tt;
        double tot=in+task+out;
        process_times[0]=tot;
        //std::cout<<"\n\nCost:"<<Cost<<std::endl;
        //std::cout<<"input_time:"<<in<<"\ntask_time:"<<task<<"\noutput_time:"<<out<<"\ntotal_time:"<<tot<<std::endl;
        std::cerr<<"\n\nstage1_input_time: "<<in<<" ms"<<"\nstage1_inference_time: "<<task+out<<" ms"<<"\nstage1_total_time: "<<tot<<" ms"<<std::endl;
    }
    void do_run_2(int core_id)
    {
        // Run graph
        //Ehsan
    	//int core_id=1;

    	if(cluster2==-1)
    		return;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core1, &set);
        //CPU_SET(1,&set);
        //CPU_SET(0,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");

        //std::cerr<<"start running second graph ...\n";
        ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
        double in2=0;
        double task2=0;
        double out2=0;
        int tt=(common_params.n);
        auto tstart=std::chrono::high_resolution_clock::now();
        //std::cout<<tstart.time_since_epoch().count()<<std::endl;
        //std::cout<<tt<<std::endl;
        for(int i=0;i<(tt+1);i++){
        	if(i==1){
        		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        		if(cluster==-1){
        			std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        			std::cerr<<"\nRunning Inference ... ";
        		}
        		tstart=std::chrono::high_resolution_clock::now();
        		//std::cout<<tstart.time_since_epoch().count()<<std::endl;
        		in2=task2=out2=0;
        	}
			if(imgs){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				std::cout<<"\n\nSecond graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				//im_acc->set_filename(images_list[image_index++]);
			}
			if(second)
			{
				graph2.run(in2,task2,out2,annotate);
			}
			if(i==1)
				in2=0;
        }
        auto tfinish=std::chrono::high_resolution_clock::now();
        //std::cout<<tfinish.time_since_epoch().count()<<std::endl;
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=1000*cost0/tt;
        //std::cout<<"\n\nCost:"<<Cost<<std::endl;
        in2=1000*in2/tt;
        task2=1000*task2/tt;
        out2=1000*out2/tt;
        double tot2=in2+task2+out2;
        process_times[1]=tot2;
        //std::cout<<"Cost:"<<Cost<<std::endl;
        //std::cout<<"\n\ninput2_time:"<<in2<<"\ntask2_time:"<<task2<<"\noutput2_time:"<<out2<<"\ntotal2_time:"<<tot2<<std::endl;
        std::cerr<<"\n\nstage2_input_time: "<<in2<<" ms"<<"\nstage2_inference_time: "<<task2+out2<<" ms"<<"\nstage2_total_time: "<<tot2<<" ms"<<std::endl;

    }
    void do_run_3(int core_id)
        {
            // Run graph
            //Ehsan
        	//int core_id=1;
    		if(cluster3==-1)
    			return;

            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(core2, &set);
            //CPU_SET(1,&set);
            //CPU_SET(0,&set);
            ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");

            //std::cerr<<"start running third graph ...\n";
            ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
            double in3=0;
            double task3=0;
            double out3=0;
            int tt=(common_params.n);
            auto tstart=std::chrono::high_resolution_clock::now();
            //std::cout<<tstart.time_since_epoch().count()<<std::endl;
            //std::cout<<tt<<std::endl;
            for(int i=0;i<(tt+1);i++){
            	if(i==1){
            		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            		if(cluster==-1 && cluster2==-1){
            			std::cerr<<"\nRunning Inference ... ";
            		}
            		tstart=std::chrono::high_resolution_clock::now();
            		//std::cout<<tstart.time_since_epoch().count()<<std::endl;
            		in3=task3=out3=0;
            	}
    			if(imgs){
    				if(image_index>=images_list.size())
    						image_index=image_index%images_list.size();
    				std::cout<<"\n\nThird graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
    				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
    				//im_acc->set_filename(images_list[image_index++]);
    			}
    			if(third)
    			{
    				graph3.run(in3,task3,out3,annotate);
    			}
    			if(i==1)
    				in3=0;
            }
            auto tfinish=std::chrono::high_resolution_clock::now();
            //std::cout<<tfinish.time_since_epoch().count()<<std::endl;
            double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
            double Cost=1000*cost0/tt;
            //std::cout<<"\n\nCost:"<<Cost<<std::endl;
            in3=1000*in3/tt;
            task3=1000*task3/tt;
            out3=1000*out3/tt;
            double tot3=in3+task3+out3;
            process_times[2]=tot3;
            //std::cout<<"Cost:"<<Cost<<std::endl;
            //std::cout<<"\n\ninput3_time:"<<in3<<"\ntask3_time:"<<task3<<"\noutput3_time:"<<out3<<"\ntotal3_time:"<<tot3<<std::endl;
            std::cerr<<"\n\nstage3_input_time: "<<in3<<" ms"<<"\nstage3_inference_time: "<<task3+out3<<" ms"<<"\nstage3_total_time: "<<tot3<<" ms"<<std::endl;
        }
	

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    CommonGraphParams  common_params2;
    Stream			   graph2;
    TensorDescriptor   input_descriptor2;
    bool     		   second=false;

    CommonGraphParams  common_params3;
    Stream			   graph3;
    TensorDescriptor   input_descriptor3;
    bool     		   third=false;

    Stream *sub_graph=&graph;
    CommonGraphParams *_common_params=&common_params;
    int Layer=0;
    int p=0;
    int p2=0;
    bool			   annotate{false};
    int gpu_index=0;
    int cluster,cluster2,cluster3 = 0;
};

/** Main program for AlexNet
 *
 * Model is based on:
 *      https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
 *      "ImageNet Classification with Deep Convolutional Neural Networks"
 *      Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E
 *
 * Provenance: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 *
 * @return Return code
 */
int main(int argc, char **argv)
{
    //Ehsan
    /*
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(4, &cpuset);
    int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);
    if(e !=0)
        std::cout << "Error in setting sched_setaffinity \n";
    */
	//init_waitqueue_head(&wq);
    return arm_compute::utils::run_example<GraphAlexnetExample>(argc, argv);
}
