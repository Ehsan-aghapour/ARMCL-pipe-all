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


std::map<int,int> gr_layer;
/** Example demonstrating how to implement AlexNet's network using the Compute Library's graph API */
class GraphAlexnetExample : public Example
{
public:
    GraphAlexnetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params()
    {
    }

    void Attach_Layer(){
    	static int P_Layer=0;
    	Layer++;
    	bool graph_finished=false;
    	if(Layer==Layers)
    		graph_finished=true;
    	else if(classes[gr_layer[Layer]]!=classes[gr_layer[Layer-1]]){
    		graph_finished=true;
    		P_Layer=Layer-1;
    	}
		if( graph_finished ){
			if(Layer!=Layers){
				if(targets[gr_layer[Layer-1]]==arm_compute::graph::Target ::CL){
					common_params.labels="transfer";
					//common_params.image="transfer";
				}
				else{
					common_params.labels="transfer_wait";
				}
				(*sub_graph)<<OutputLayer(get_Sender_accessor(common_params, gr_layer[Layer]));
			}

			GraphConfig config;
			if(classes[gr_layer[Layer-1]]==0){
				config.num_threads = common_params.threads2;
				//config.cluster=0;
			}
			else{
				config.num_threads = common_params.threads;
				//config.cluster=1;
			}
			config.cluster=classes[gr_layer[Layer-1]];
			config.use_tuner   = common_params.enable_tuner;
			config.tuner_mode  = common_params.tuner_mode;
			config.tuner_file  = common_params.tuner_file;
			config.mlgo_file   = common_params.mlgo_file;
			//std::cout<<"Finalizing graph_"<<gr_layer[Layer-1]<<"\t after Layer:"<<Layer-1<<std::endl;
			//std::cout<<"class:"<<config.cluster<<"\t target:"<<int(targets[gr_layer[Layer-1]])<<'='<<int(common_params.target)<<std::endl;
			sub_graph->finalize(common_params.target, config);
			if(gr_layer[Layer-1]>0){
				for(auto &node : sub_graph->graph().nodes())
				{
					if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
					{
						//PrintThread{}<<"adding rec "<<Layer<<std::endl;
						Receivers.push_back(node->output(0));
						continue;
					}
				}
			}
			std::string l;
			if(Layer!=Layers){
				for(auto &node : sub_graph->graph().nodes())
				{
					if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
					{
						Transmitters.push_back(node->input(0));
						continue;
					}
				}
				sub_graph=(graphs[gr_layer[Layer]]);
				if(classes[gr_layer[Layer-1]]==2){
					common_params.image="transfer_wait";
				}
				else{
					common_params.image="transfer";
				}
				common_params.target=targets[gr_layer[Layer]];
				const auto        operation_layout = common_params.data_layout;
				TensorDescriptor input_descriptor = TensorDescriptor(Transmitters[gr_layer[Layer-1]]->desc().shape, common_params.data_type).set_layout(operation_layout);
				(*sub_graph) << common_params.target
				              << common_params.fast_math_hint;
				(*sub_graph)<<InputLayer(input_descriptor, get_Receiver_accessor(common_params,gr_layer[Layer-1]));
				cpu_set_t set;
				CPU_ZERO(&set);
				CPU_SET(core[classes[gr_layer[Layer]]],&set);
				ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
			}


		}
    }


    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);
        //common_params2 = consume_common_graph_parameters(common_opts);
        
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

        std::string lbl=common_params.labels;
        std::string order=common_params.order;
        Layers=order.size();
        int g=0;
        for(int i=0;i<Layers;i++){
        	if(i==0){
        		//Stream graph(i,"AlexNet");
				if (order[i]=='B'){
					targets.push_back(arm_compute::graph::Target ::NEON);
					classes.push_back(1);
				}
				if (order[i]=='L'){
					targets.push_back(arm_compute::graph::Target ::NEON);
					classes.push_back(0);
				}
				if (order[i]=='G'){
					targets.push_back(arm_compute::graph::Target ::CL);
					classes.push_back(2);
				}
				graphs.push_back(new Stream(g,"AlexNet"));
				gr_layer[i]=g;
        	}

        	else if (order[i]!=order[i-1]){
        		//Stream graph(i,"AlexNet");
				if (order[i]=='B'){
					targets.push_back(arm_compute::graph::Target ::NEON);
					classes.push_back(1);
				}
				if (order[i]=='L'){
					targets.push_back(arm_compute::graph::Target ::NEON);
					classes.push_back(0);
				}
				if (order[i]=='G'){
					targets.push_back(arm_compute::graph::Target ::CL);
					classes.push_back(2);
				}
				g++;
				graphs.push_back(new Stream(g,"AlexNet"));
				gr_layer[i]=g;
        	}

        	else{
        		gr_layer[i]=g;
        	}
        }
        /*for(int i=0;i<8;i++){
        	std::cout<<"Layer:"<<i<<'\t'<<"graph:"<<gr_layer[i]<<'\t'<<"class:"<<classes[gr_layer[i]]<<'\t'<<"target:"<<int(targets[gr_layer[i]])<<std::endl;
        }*/
        //std::vector<Stream> _graphs(g);
        //graphs.swap(_graphs);
        //static std::vector<std::mutex> _mx(graphs.size()-1);
        //mx.swap(_mx);
        //static std::vector<std::condition_variable> _cvs(graphs.size()-1);
        //cvs.swap(_cvs);
        //std::vector<arm_compute::Tensor> _buffer_tensors(graphs.size()-1);
       // std::cout<<"heyyyy:"<<_buffer_tensors.size()<<std::endl;
        //std::string mm;
        //std::cin>>mm;
        //buffer_tensors.swap(_buffer_tensors);
        //std::cout<<"heyyyy2:"<<buffer_tensors.size()<<std::endl;
               // std::string mm;
          //      std::cin>>mm;

        cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(core[classes[gr_layer[Layer]]],&set);
		ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");

        std::cout << common_params << std::endl;
        sub_graph=(graphs[gr_layer[Layer]]);
        annotate=common_params.annotate;
        ann=annotate;
        save_model=common_params.save;
        common_params.target=targets[gr_layer[Layer]];
        (*sub_graph) << common_params.target
              << common_params.fast_math_hint;
        		auto ii=InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)));
        		(*sub_graph)  << ii;
              // Layer 1
		        		(*sub_graph) << ConvolutionLayer(
                  11U, 11U, 96U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
                  PadStrideInfo(4, 4, 0, 0))
              .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool1");

        Attach_Layer();

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

		Attach_Layer();

		// Layer 3
		(*sub_graph)<< ConvolutionLayer(
						  3U, 3U, 384U,
						  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy", weights_layout),
						  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"),
						  PadStrideInfo(1, 1, 1, 1))
					  .set_name("conv3")
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3");

		Attach_Layer();

		// Layer 4
		(*sub_graph)<< ConvolutionLayer(
			3U, 3U, 384U,
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy", weights_layout),
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"),
			PadStrideInfo(1, 1, 1, 1), 2)
		.set_name("conv4")
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4");

		Attach_Layer();

		// Layer 5
		(*sub_graph)<< ConvolutionLayer(
			3U, 3U, 256U,
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy", weights_layout),
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"),
			PadStrideInfo(1, 1, 1, 1), 2)
		.set_name("conv5")
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5")
		<< PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool5");

		Attach_Layer();

		// Layer 6
		(*sub_graph)<< FullyConnectedLayer(
			4096U,
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy", weights_layout),
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy"))
		.set_name("fc6")
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu6");

		Attach_Layer();

		// Layer 7
		(*sub_graph)<< FullyConnectedLayer(
			4096U,
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy", weights_layout),
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy"))
		.set_name("fc7")
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu7");

		Attach_Layer();
		common_params.labels=lbl;

		// Layer 8
		(*sub_graph)<< FullyConnectedLayer(
			1000U,
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy", weights_layout),
			get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy"))
		.set_name("fc8")
		// Softmax
		<< SoftmaxLayer().set_name("prob")
		<< OutputLayer(get_output_accessor(common_params, 5));

		Attach_Layer();

		im_acc=dynamic_cast<ImageAccessor*>(graphs[0]->graph().node(0)->output(0)->accessor());

		std::cout<<"Total layers:"<<Layer<<std::endl<<std::endl;

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

    	std::vector<std::thread*> stages;
    	for(int i=0;i<graphs.size();i++){
    		stages.push_back(new std::thread(&GraphAlexnetExample::run,this,i));
    		//std::cout<<"thread "<< i<<" created\n";
    		//stages[i]->join();
    	}
    	for(int i=0;i<stages.size();i++){
			stages[i]->join();
    	}
    	del();

    }
    void run(int graph_id){
    	int cl=classes[graph_id];
    	int core_id=core[cl];
    	cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(core_id,&set);
		ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
		PrintThread{}<<"start running graph "<<graph_id<<std::flush<<std::endl;
		double in=0;
		double task=0;
		double out=0;
		int n=(common_params.n);
		auto tstart=std::chrono::high_resolution_clock::now();
		for(int i=0;i<(n+1);i++){
			if(i==1){
				in=task=out=0;
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				tstart=std::chrono::high_resolution_clock::now();
				//std::cout<<tstart.time_since_epoch().count()<<std::endl;
			}
			if(imgs && graph_id==0){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				PrintThread{}<<"\n\nFirst graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				im_acc->set_filename(images_list[image_index++]);
			}
			graphs[graph_id]->run(in,task,out,annotate);
		}
		auto tfinish=std::chrono::high_resolution_clock::now();
		double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		double Cost=cost0/n;
		in=in/n;
		task=task/n;
		out=out/n;
		double tot=in+task+out;
		PrintThread{}<<"\n\nCost"<<graph_id<<":"<<Cost<<std::endl;
		PrintThread{}<<"input"<<graph_id<<"_time:"<<in<<"\ntask"<<graph_id<<"_time:"<<task<<"\noutput"<<graph_id<<"_time:"<<out<<"\ntotal"<<graph_id<<"_time:"<<tot<<std::endl;

    }

	

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;

    CommonGraphParams  common_params;
    std::vector<Stream*> graphs;
    std::vector<arm_compute::graph::Target> targets;
    std::vector<int> classes;
    std::vector<TensorDescriptor> inputs;
    Stream *sub_graph=NULL;
    int Layer=0;
    int Layers=0;
    bool			   annotate{false};
    std::map<int, int> core = {{0, 1}, {1, 5}, {2, 4}};
    ImageAccessor *im_acc=NULL;
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
    return arm_compute::utils::run_example<GraphAlexnetExample>(argc, argv);
}
