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


//Ehsan
#include<chrono>
#include <sys/types.h>
#include <dirent.h>

#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;


//std::map<int,int> core;
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
std::set<int> squeeze_blocking {1,2,5,6,9,10,14,15,18,19,22,23,26,27,31,32,35,37,39};
int end_tasks[]={1,2,5,6,9,10,14,15,18,19,22,23,26,27,31,32,35,37,39};

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API */
class GraphSqueezenetExample : public Example
{
public:
    GraphSqueezenetExample()
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
				//sub_graph->finalize(common_params.target, config, &squeeze_blocking,common_params.layer_time);
				sub_graph->finalize(common_params.target, config, &e_t,common_params.layer_time);
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


    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);


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

        // Print parameter values
        //std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<CaffePreproccessor>(mean_rgb);

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, operation_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;


        //Ehsan
        //**********************************************************************************

        int n_l=19;
        std::cerr<<"Number of Layers: "<<n_l<<std::endl;
        std::string lbl=common_params.labels;
        if(common_params.order.size()==1){
        	common_params.order=std::string(n_l, common_params.order[0]);
        }
        if(common_params.order[1]=='-'){
        	common_params.order=std::string(common_params.partition_point,common_params.order[0])+
        			std::string(common_params.partition_point2-common_params.partition_point,common_params.order[2])+
					std::string(n_l-common_params.partition_point2,common_params.order[4]);
        }
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
					graphs.push_back(new Stream(g,"AlexNet"));
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

					graphs.push_back(new Stream(g+1,"AlexNet"));
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

        //***************************************************************


        (*sub_graph) << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)));
        (*sub_graph)<< ConvolutionLayer(
                  7U, 7U, 96U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_b.npy"),
                  PadStrideInfo(2, 2, 0, 0))
              .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu_conv1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool1");

        Attach_Layer();

        (*sub_graph)<< ConvolutionLayer(
                  1U, 1U, 16U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire2/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire2/relu_squeeze1x1");

        Attach_Layer();

        (*sub_graph) << get_expand_fire_node(data_path, "fire2", weights_layout, 64U, 64U).set_name("fire2/concat");

        Attach_Layer();

        (*sub_graph) << ConvolutionLayer(
                  1U, 1U, 16U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire3/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire3/relu_squeeze1x1");

        Attach_Layer();

        (*sub_graph) << get_expand_fire_node(data_path, "fire3", weights_layout, 64U, 64U).set_name("fire3/concat");

        Attach_Layer();

        (*sub_graph) << ConvolutionLayer(
                  1U, 1U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire4/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire4/relu_squeeze1x1");

        Attach_Layer();

        (*sub_graph) << get_expand_fire_node(data_path, "fire4", weights_layout, 128U, 128U).set_name("fire4/concat");
        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool4");

        Attach_Layer();

        (*sub_graph)<< ConvolutionLayer(
                  1U, 1U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire5/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire5/relu_squeeze1x1");

        Attach_Layer();

        (*sub_graph) << get_expand_fire_node(data_path, "fire5", weights_layout, 128U, 128U).set_name("fire5/concat");

        Attach_Layer();

        (*sub_graph) << ConvolutionLayer(
                  1U, 1U, 48U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire6/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire6/relu_squeeze1x1");

        Attach_Layer();

        (*sub_graph) << get_expand_fire_node(data_path, "fire6", weights_layout, 192U, 192U).set_name("fire6/concat");

        Attach_Layer();

        (*sub_graph) << ConvolutionLayer(
                  1U, 1U, 48U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire7/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire7/relu_squeeze1x1");

        Attach_Layer();

        (*sub_graph) << get_expand_fire_node(data_path, "fire7", weights_layout, 192U, 192U).set_name("fire7/concat");

        Attach_Layer();

        (*sub_graph) << ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire8/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire8/relu_squeeze1x1");

        Attach_Layer();

        (*sub_graph) << get_expand_fire_node(data_path, "fire8", weights_layout, 256U, 256U).set_name("fire8/concat");
        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool8");

        Attach_Layer();

        (*sub_graph)<< ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire9/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire9/relu_squeeze1x1");

        Attach_Layer();

        (*sub_graph) << get_expand_fire_node(data_path, "fire9", weights_layout, 256U, 256U).set_name("fire9/concat");

        Attach_Layer();

        (*sub_graph) << ConvolutionLayer(
                  1U, 1U, 1000U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv10")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu_conv10")
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, operation_layout)).set_name("pool10");

        Attach_Layer();
        common_params.labels=lbl;

	    (*sub_graph)<< FlattenLayer().set_name("flatten")
              << SoftmaxLayer().set_name("prob")
              << OutputLayer(get_output_accessor(common_params, 5));

		Attach_Layer();

		im_acc=dynamic_cast<ImageAccessor*>(graphs[0]->graph().node(0)->output(0)->accessor());

		std::cout<<"Total layers:"<<Layer<<std::endl<<std::endl;

		return true;
    }

    void do_run() override
    {
        // Run graph
        //Ehsan
    	std::string t;
    	std::vector<std::thread*> stages;
    	int n=common_params.n;
    	for(int i=0;i<graphs.size();i++){
    		stages.push_back(new std::thread(&GraphSqueezenetExample::run,this,i));
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
			graphs[graph_id]->run(annotate,n);
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
				//std::cerr<<i<<" graph_id:"<<graph_id<<"   time:"<<std::chrono::high_resolution_clock::now().time_since_epoch().count()<<std::endl;

				graphs[graph_id]->run(annotate,n);
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

    ConcatLayer get_expand_fire_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     unsigned int expand1_filt, unsigned int expand3_filt)
    {
        std::string total_path = "/cnn_data/squeezenet_v1.0_model/" + param_path + "_";
        SubStream   i_a(*sub_graph);
        i_a << ConvolutionLayer(
                1U, 1U, expand1_filt,
                get_weights_accessor(data_path, total_path + "expand1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "expand1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/expand1x1")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_expand1x1");

        SubStream i_b(*sub_graph);
        i_b << ConvolutionLayer(
                3U, 3U, expand3_filt,
                get_weights_accessor(data_path, total_path + "expand3x3_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "expand3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/expand3x3")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_expand3x3");

        return ConcatLayer(std::move(i_a), std::move(i_b));
    }
};

/** Main program for Squeezenet v1.0
 *
 * Model is based on:
 *      https://arxiv.org/abs/1602.07360
 *      "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
 *      Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
 *
 * Provenance: https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphSqueezenetExample>(argc, argv);
}
