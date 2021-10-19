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
//bool ann=0;
std::set<int> alex_blocking {2,7,8,11,15,16,17,19};
int end_tasks[]={2,7,8,11,15,16,17,19};



//static std::mutex inout;



/** Example demonstrating how to implement AlexNet's network using the Compute Library's graph API */
class GraphAlexnetExample : public Example
{
public:
    GraphAlexnetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params()
    {
    }

    void Attach_Layer(){
    	//std::cerr<<"attaching layer "<<Layer<<" on graph:"<<gr_layer[Layer]<<std::endl;
    	static int start_Layer=0;
    	static int end_Layer=0;
    	Layer++;
    	bool graph_finished=false;
    	if(Layer==Layers)
    		graph_finished=true;
    	//else if(classes[gr_layer[Layer]]!=classes[gr_layer[Layer-1]]){
    	else if(gr_layer[Layer]!=gr_layer[Layer-1]){
    		graph_finished=true;
    		//P_Layer=Layer-1;
    	}
    	//std::cerr<<common_params.order[Layer-1]<<", finish: "<<graph_finished<<std::endl;
		if( graph_finished){
			end_Layer=Layer-1;
			if(gr_layer[Layer-1]!=-1){
				if(Layer!=Layers){
					if(targets[gr_layer[Layer-1]]==arm_compute::graph::Target ::CL){
						common_params.labels="transfer";
						//common_params.image="transfer";
					}
					else{
						common_params.labels="transfer_wait";
					}
					if(gr_layer[Layer]==-1)
						common_params.labels="";
					//(*sub_graph)<<OutputLayer(get_Sender_accessor(common_params, gr_layer[Layer-1]+1));
					(*sub_graph)<<OutputLayer(get_Sender_accessor(common_params, Transmitters.size()+1));
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
				std::set<int> e_t;
				int offset=0;
				if(start_Layer>0)
					offset=end_tasks[start_Layer-1]+1;
				for(int i=start_Layer;i<=end_Layer;i++){
					e_t.insert(end_tasks[i]-offset);
				}
				std::cout<<"Start_Layer:"<<start_Layer<<" \t End layer:"<<end_Layer<<"\n set:";
				for (auto itr = e_t.begin(); itr != e_t.end(); itr++)
				{
					std::cout << *itr<<" ";
				}
				std::cout<<std::endl;
				sub_graph->finalize(common_params.target, config, &e_t,common_params.layer_time);
				//sub_graph->finalize(common_params.target, config, &alex_blocking,common_params.layer_time);
				if(gr_layer[Layer-1]>0){
					for(auto &node : sub_graph->graph().nodes())
					{
						if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
						{
							//PrintThread{}<<"adding rec "<<Layer<<std::endl;
							if(common_params.image!=""){
								Receivers.push_back(node->output(0));
								continue;
							}
						}
					}
				}
			}

			else if(Layer!=Layers){
				common_params.labels="";
				(*sub_graph)<<OutputLayer(get_Sender_accessor(common_params, gr_layer[Layer]));
			}

			if(Layer!=Layers){
				arm_compute::graph::Tensor* temp_sender;
				TensorShape tshape;
				//if(gr_layer[Layer]!=-1){
					for(auto &node : sub_graph->graph().nodes())
					{
						if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
						{
							if(gr_layer[Layer-1]!=-1 && gr_layer[Layer]!=-1){
								Transmitters.push_back(node->input(0));
								//tshape=Transmitters[gr_layer[Layer-1]]->desc().shape;
								tshape=Transmitters[Transmitters.size()-1]->desc().shape;
							}
							else{
								temp_sender=node->input(0);
								tshape=temp_sender->desc().shape;
							}
							continue;
						}
					}

				if(gr_layer[Layer]!=-1){
					sub_graph=(graphs[gr_layer[Layer]]);

					if(gr_layer[Layer-1]==-1){
						common_params.image="";
					}
					else{
						if(classes[gr_layer[Layer-1]]==2){
							common_params.image="transfer_wait";
						}
						else{
							common_params.image="transfer";
						}
					}

					common_params.target=targets[gr_layer[Layer]];
					const auto        operation_layout = common_params.data_layout;
					TensorDescriptor input_descriptor = TensorDescriptor(tshape, common_params.data_type).set_layout(operation_layout);
					(*sub_graph) << common_params.target
								  << common_params.fast_math_hint;
					//std::cout<<common_params.image<<", "<<Transmitters.size()-1<<std::endl;

					//auto tt=InputLayer(input_descriptor, get_Receiver_accessor(common_params,gr_layer[Layer]-1));
					//auto tt=InputLayer(input_descriptor, get_Receiver_accessor(common_params,Transmitters.size()-1));

					(*sub_graph)<<InputLayer(input_descriptor, get_Receiver_accessor(common_params,Transmitters.size()-1));

					cpu_set_t set;
					CPU_ZERO(&set);
					CPU_SET(core[classes[gr_layer[Layer]]],&set);
					ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
				}
				else{
					delete dump_graph;
					dump_graph=new Stream(1000,"AlexNet");
					sub_graph=dump_graph;
					common_params.target=arm_compute::graph::Target ::NEON;
					const auto        operation_layout = common_params.data_layout;
					TensorDescriptor input_descriptor = TensorDescriptor(tshape, common_params.data_type).set_layout(operation_layout);
					(*sub_graph) << common_params.target
								  << common_params.fast_math_hint;
					common_params.image="";
					(*sub_graph)<<InputLayer(input_descriptor, get_Receiver_accessor(common_params,gr_layer[Layer]-1));
				}
			}
			start_Layer=Layer;
		}
		//std::cerr<<"Attached\n";
    }

    TensorShape output_shape(Stream* g){
    	arm_compute::graph::Tensor* temp_sender;
		TensorShape tshape;
		//if(gr_layer[Layer]!=-1){

		for(auto &node : sub_graph->graph().nodes())
		{
			std::cout<<"Node name: "<<node.get()->name()<<" \t output shape: "<<node.get()->output(0)->desc().shape<<std::endl<<std::flush;
			/*if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
			{
				temp_sender=node->input(0);
				tshape=temp_sender->desc().shape;
				std::cerr<<"Node name: "<<node.get()->name();
				continue;
			}*/
		}

		auto *last_node=&(sub_graph->graph().nodes()[sub_graph->tail_node()]);
		tshape=last_node[0]->output(0)->desc().shape;
		std::cout<<"\nLast node name:"<<last_node[0]->name()<<" \t shape:"<<tshape<<std::endl<<std::endl<<std::flush;

		return tshape;
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
        const TensorShape tensor_shape     = permute_shape(TensorShape(common_params.input_s, common_params.input_s, common_params.input_c, 1U), DataLayout::NCHW, operation_layout);
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
				if (order[i]!='-'){
					graphs.push_back(new Stream(g+1,"AlexNet"));
					gr_layer[i]=g;
				}
				if(order[i]=='-'){
					gr_layer[i]=-1;
				}
        	}

        	else if (order[i]!=order[i-1]){
        		//Stream graph(i,"AlexNet");
        		if(order[i]=='-'){
        			gr_layer[i]=-1;
        		}
        		else{
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
					gr_layer[i]=graphs.size()-1;
					g=graphs.size()-1;
        		}

        	}

        	else{
        		if(order[i]!='-')
        			gr_layer[i]=g;
        		else
        			gr_layer[i]=-1;
        	}
        }
        for(int i=0;i<Layers;i++){
        	//std::cerr<<i<<"\t"<<gr_layer[i]<<std::endl;
        	if(order[i]=='-'){
        		dump_graph=new Stream(1000,"AlexNEt");
        		break;
        	}
        }
        per_frame=(graphs.size()>1);
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

        annotate=common_params.annotate;
		//ann=annotate;
		save_model=common_params.save;

        if(gr_layer[Layer]==-1){
        	sub_graph=dump_graph;
        	common_params.target=arm_compute::graph::Target ::NEON;
        }
        else{
        	sub_graph=(graphs[gr_layer[Layer]]);
        	common_params.target=targets[gr_layer[Layer]];
        }

        (*sub_graph) << common_params.target
              << common_params.fast_math_hint;
        		auto ii=InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)));
        		(*sub_graph)  << ii;
        		std::cout<<"stride:"<<common_params.stride<<std::endl;
              // Layer 1
		      (*sub_graph) << ConvolutionLayer(
		    		  common_params.kernel_s, common_params.kernel_s, common_params.kernel_c,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
                  PadStrideInfo(common_params.stride, common_params.stride, 0, 0))
              .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool1");

        Attach_Layer();
        //auto tt=output_shape(sub_graph);
        //return true;
		common_params.labels=lbl;

		// Layer 8
		(*sub_graph)<< FullyConnectedLayer(
			10U,
			get_weights_accessor(data_path, " ", weights_layout),
			get_weights_accessor(data_path, " "))
		.set_name("fc2")
		// Softmax
		<< SoftmaxLayer().set_name("prob")
		<< OutputLayer(get_output_accessor(common_params, 5));

		Attach_Layer();

		im_acc=dynamic_cast<ImageAccessor*>(graphs[0]->graph().node(0)->output(0)->accessor());

		std::cout<<"Total layers:"<<Layer<<std::endl<<std::endl<<std::flush;

        /*

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
		*/

		// Save the opencl kernels to a file
		/*if(common_opts.enable_cl_cache)
		{
		#ifdef ARM_COMPUTE_CL
			save_program_cache_to_file();
		#endif // ARM_COMPUTE_CL
		}*/

		return true;
    }

    void do_run() override
    {
        // Run graph
        //Ehsan
    	//return;
    	cpu_set_t set;
    	CPU_ZERO(&set);
    	CPU_SET(1,&set);
    	ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
    	//std::string t;
    	std::vector<std::thread*> stages;
    	int n=common_params.n;
    	for(int i=0;i<graphs.size();i++){
    		stages.push_back(new std::thread(&GraphAlexnetExample::run,this,i));
    		//std::cout<<"thread "<< i<<" created\n";
    		//stages[i]->join();
    	}
    	for(int i=0;i<stages.size();i++){
			stages[i]->join();
    	}
    	for(int i=0;i<graphs.size();i++){
			//std::cout<<"graph_id: "<<i<<" \t start: "<<graphs[i]->get_start_time().time_since_epoch().count()<<" \t end: "<<graphs[i]->get_finish_time().time_since_epoch().count()<<std::endl;
    		if(common_params.layer_time)
    				graphs[i]->measure(n);

			double tot=graphs[i]->get_input_time()+graphs[i]->get_task_time()+graphs[i]->get_output_time();
			PrintThread{}<<"\n\nCost"<<i<<":"<<1000*graphs[i]->get_cost_time()/n<<std::endl;
			PrintThread{}<<"input"<<i<<"_time:"<<1000*graphs[i]->get_input_time()/n<<"\ntask"<<i<<"_time:"<<1000*graphs[i]->get_task_time()/n<<"\noutput"<<i<<"_time:"<<1000*graphs[i]->get_output_time()/n<<"\ntotal"<<i<<"_time:"<<1000*tot/n<<std::endl;
			std::cout<<"***************************************\n\n";

		}


    	std::cout<<"Frame Latency: "<<1000*latency/(common_params.n)<<std::endl;
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
		bool layer_timing=common_params.layer_time;
		bool end=(graph_id==graphs.size()-1);
		latency=0;
		//auto tstart=std::chrono::high_resolution_clock::now();
		//std::cerr<<"graph__id:"<<graph_id<<"   time:"<<tstart.time_since_epoch().count()<<std::endl;
		if(imgs && graph_id==0){
			if(image_index>=images_list.size())
					image_index=image_index%images_list.size();
			PrintThread{}<<"\n\nFirst graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
			//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
			im_acc->set_filename(images_list[image_index++]);
		}
		if(layer_timing){
			//std::cerr<<i<<" graph_id:"<<graph_id<<"   time:"<<std::chrono::high_resolution_clock::now().time_since_epoch().count()<<std::endl;
			graphs[graph_id]->run(annotate);
			//graphs[graph_id]->set_finish_time(std::chrono::high_resolution_clock::now());
		}
		else{
			graphs[graph_id]->run(annotate);
		}

		graphs[graph_id]->set_input_time(0);
		graphs[graph_id]->set_task_time(0);
		graphs[graph_id]->set_output_time(0);
		graphs[graph_id]->set_cost_time(0);
		if(layer_timing)
			graphs[graph_id]->reset();
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		if(graph_id==0){
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}
		auto tstart=std::chrono::high_resolution_clock::now();

		//std::cout<<tstart.time_since_epoch().count()<<std::endl;
		if(graph_id==0)
			start=std::chrono::high_resolution_clock::now();
		for(int i=0;i<n;i++){
			if(imgs && graph_id==0){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				PrintThread{}<<"\n\nFirst graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				im_acc->set_filename(images_list[image_index++]);
			}
			if(layer_timing){
				//std::cerr<<i<<" graph_id:"<<graph_id<<" start  time:"<<(std::chrono::high_resolution_clock::now().time_since_epoch().count()/1000000)%10000<<std::endl;
				graphs[graph_id]->run(annotate,n);
				//std::cerr<<i<<" graph_id:"<<graph_id<<"  finish time:"<<(std::chrono::high_resolution_clock::now().time_since_epoch().count()/1000000)%10000<<std::endl;
				//graphs[graph_id]->set_finish_time(std::chrono::high_resolution_clock::now());
				if(end){
					//latency += std::chrono::duration_cast<std::chrono::duration<double>>(graphs[graph_id]->get_finish_time() - graphs[0]->get_start_time()).count();
					latency += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
					start=std::chrono::high_resolution_clock::now();
				}
			}
			else{
				graphs[graph_id]->run(annotate);
			}
		}
		auto tfinish=std::chrono::high_resolution_clock::now();
		double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		//graphs[graph_id]->set_input_time(in);
		//graphs[graph_id]->set_task_time(task);
		//graphs[graph_id]->set_output_time(out);
		graphs[graph_id]->set_cost_time(cost0);
		/*double Cost=cost0/n;
		in=in/n;
		task=task/n;
		out=out/n;
		double tot=in+task+out;
		PrintThread{}<<"\n\nCost"<<graph_id<<":"<<Cost<<std::endl;
		PrintThread{}<<"input"<<graph_id<<"_time:"<<in<<"\ntask"<<graph_id<<"_time:"<<task<<"\noutput"<<graph_id<<"_time:"<<out<<"\ntotal"<<graph_id<<"_time:"<<tot<<std::endl;
		std::cout<<"***************************************\n\n";*/


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
    Stream *dump_graph=NULL;
    std::map<int,int> gr_layer;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    //std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    double latency=0;
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
