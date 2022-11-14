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
#include "utils/GraphUtils_NPU.h"
#include "utils/Utils.h"

//Power
#include "power.h"
#define Power_Measurement 1

//NPU
#include "rknn_api.h"
#include "rockx.h"

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
        if(arm_compute::utility::endswith(dp->d_name, ".ppm") || arm_compute::utility::endswith(dp->d_name, ".jpg"))
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
int qend_tasks[]={1,2,5,6,9,10,14,15,18,19,22,23,26,27,31,32,35,37,39};

//NPU
arm_compute::ITensor *temp_tensor=NULL;
arm_compute::Tensor Input_tensor;

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API */
class GraphSqueezenetExample : public Example
{
public:
    GraphSqueezenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params()
    {
    }

    cpu_set_t* set_cores(cpu_set_t *set,bool _one_master_core, int _core){
		if(one_master_core){
			CPU_SET(_core,set);
		}
		else{
			if(_core < common_params.little_cores){
				for(int i=0;i<common_params.little_cores;i++){
					CPU_SET(i,set);
				}
			}
			else{
				for(int i=common_params.little_cores;i<common_params.total_cores;i++){
					CPU_SET(i,set);
				}
			}
		}
		return set;
	}

    void fill_host_core(){
    	std::vector<int> little_cores;
		std::vector<int> big_cores;
		for(int i=0;i<common_params.little_cores;i++){
			little_cores.push_back(i);
		}
		for(int i=common_params.little_cores; i<common_params.total_cores;i++){
			big_cores.push_back(i);
		}


		char NPU_h=common_params.npu_host;
		char GPU_h=common_params.gpu_host;

		host_core[0]=little_cores.back();
		if( little_cores.size() > 1){
			little_cores.pop_back();
		}
		host_core[1]=big_cores.back();
		if( big_cores.size() > 1){
			big_cores.pop_back();
		}
		if(GPU_h=='B'){
			host_core[2]=big_cores.back();
			if( big_cores.size() > 1){
				big_cores.pop_back();
			}
		}
		if(GPU_h=='L'){
			host_core[2]=little_cores.back();
			if (little_cores.size() >1 ){
				little_cores.pop_back();
			}
		}
		if(NPU_h=='B'){
			npu_host=big_cores.back();
			if( big_cores.size() > 1){
				big_cores.pop_back();
			}
		}
		if(NPU_h=='L'){
			npu_host=little_cores.back();
			if (little_cores.size() >1 ){
				little_cores.pop_back();
			}
		}
		std::cout<<"NPU host: "
				<<npu_host
				<<"\nLittle Cores host: "
				<<host_core[0]
				<<"\nBig Cores host: "
				<<host_core[1]
				<<"\nGPU host: "
				<<host_core[2]
				<<std::endl;
    }

    int npu_init_context(int npu_index){
        	//std::string model_name="/data/data/com.termux/files/home/ARMCL-RockPi/graphs/"+NPU_Model_Name[npu_index];
        	std::string model_name="/data/data/com.termux/files/home/ARMCL-RockPi/graphs/"+NPU_Model_Name;
        	//std::string mm="deploy.rknn";
        	//std::string model_name="/data/data/com.termux/files/home/ARMCL-RockPi/graphs/"+mm;
    #if NPU_Debug
        	std::cerr<<"Model name: "<<"/data/data/com.termux/files/home/ARMCL-RockPi/graphs/"+NPU_Model_Name<<std::endl;
        	std::cerr<<"npu index:"<<npu_index<<" len npu_contexts: "<<NPU_Contexts.size()<<std::endl;
    #endif
        	FILE *fp = fopen(model_name.c_str(), "rb");
    		if(fp == NULL) {
    			printf("fopen %s fail!\n", model_name.c_str());
    			return -1;
    		}
    		fseek(fp, 0, SEEK_END);
    		int model_len = ftell(fp);
    		void *model = malloc(model_len);
    		fseek(fp, 0, SEEK_SET);
    		if(model_len != fread(model, 1, model_len, fp)) {
    			printf("fread %s fail!\n", model_name.c_str());
    			free(model);
    			fclose(fp);
    			return -1;
    		}
    		std::cerr<<"model reading done.\n";
    		fclose(fp);

    		// RKNN_FLAG_ASYNC_MASK: enable async mode to use NPU efficiently.
    		//int ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_PRIOR_MEDIUM|RKNN_FLAG_ASYNC_MASK);
    		int ret = rknn_init(&NPU_Contexts[npu_index], model, model_len, RKNN_FLAG_PRIOR_MEDIUM);
    		std::cerr<<"NPU Initialized\n";
    		//int ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_COLLECT_PERF_MASK);
    		free(model);

    		if(ret < 0) {
    			printf("rknn_init fail! ret=%d\n", ret);
    			return -1;
    		}
    		return 0;
        }

        void Attach_Layer(){
    #if NPU_Debug
        	std::cerr<<"\n\n\n\n*************************** attaching layer "<<Layer<<" on graph:"<<gr_layer[Layer]<<std::endl;
        	/*for(int i=0;i<NPU_Receivers.size();i++){
    			std::cerr<<"att:\nNPU Rec: "<<i<<" Con id: "<<NPU_Receivers[i]->get_connection_id()<<'\n';
    		}*/
    #endif
        	//for all connection indexing(normal + npu):
        	static int Connection_id=0;
        	//for normal connection just(normal):
        	static int T_id=0;
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

        	//(0) if subgraph finished ************
    		if( graph_finished){
    #if NPU_Debug
    		std::cerr<<"SubGraph finished\n";
    #endif
    			end_Layer=Layer-1;
    			//(1)If it is not (dummy or NPU) *****************************************
    			//if(gr_layer[Layer-1]!=-1){
    			if(gr_layer[Layer-1]>=0){

    #if NPU_Debug
    				std::cerr<<"Attach layer: finalizing normal subgraph\n";
    #endif
    				//(1-0)if it is not last sungraph --> setup common_params.labels *****
    				if(Layer!=Layers){
    					if(targets[gr_layer[Layer-1]]==arm_compute::graph::Target ::CL){
    						common_params.labels="transfer";
    						//common_params.image="transfer";
    					}
    					else{
    						common_params.labels="transfer_wait";
    					}
    					//Her should consider npu as next subgraph
    					if(gr_layer[Layer]==-1)
    						common_params.labels="";
    					//NPU:
    					if(gr_layer[Layer]==-2){
    						common_params.labels="transfer_to_npu";
    					}
    					//(*sub_graph)<<OutputLayer(get_Sender_accessor(common_params, gr_layer[Layer-1]+1));

    #if NPU_Debug
    		std::cerr<<"label for subgraph: "<<common_params.labels<<std::endl;
    #endif
    					//(*sub_graph)<<OutputLayer(get_Sender_accessor(common_params, Transmitters.size()+1));
    					(*sub_graph)<<OutputLayer(get_Sender_accessor(common_params, Connection_id, T_id));
    				}
    				//(1-0)End of it is not last subgraph *****

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

    				//quant
    				config.convert_to_uint8 = (common_params.data_type == DataType::QASYMM8);
    				if(common_params.data_type == DataType::QASYMM8){
    					memcpy(end_tasks,qend_tasks,sizeof(end_tasks));
    				}
    				//std::cout<<"Finalizing graph_"<<gr_layer[Layer-1]<<"\t after Layer:"<<Layer-1<<std::endl;
    				//std::cout<<"class:"<<config.cluster<<"\t target:"<<int(targets[gr_layer[Layer-1]])<<'='<<int(common_params.target)<<std::endl;
    				std::set<int> e_t;
    				int offset=0;
    				if(start_Layer>0)
    					offset=end_tasks[start_Layer-1]+1;
    				for(int i=start_Layer;i<=end_Layer;i++){
    					e_t.insert(end_tasks[i]-offset);
    				}
    #if NPU_Debug
    				std::cerr<<"Start_Layer:"<<start_Layer<<" \t End layer:"<<end_Layer<<"\n set:";
    				for (auto itr = e_t.begin(); itr != e_t.end(); itr++)
    				{
    					std::cerr << *itr<<" ";
    				}
    				std::cerr<<std::endl;
    #endif
    				sub_graph->finalize(common_params.target, config, &e_t,common_params.layer_time);
    				if(gr_layer[Layer-1]>0){
    					for(auto &node : sub_graph->graph().nodes())
    					{
    						if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
    						{
    							//PrintThread{}<<"adding rec "<<Layer<<std::endl;
    							//HERE:if(common_params.image!=""){
    							if(common_params.image!="" && common_params.image!="transfer_from_npu"){
    								Receivers.push_back(node->output(0));
    								continue;
    							}
    						}
    					}
    				}
    #if NPU_Debug
    				std::cerr<<"Attach layer: normal subgraph finalized\n";
    #endif
    			}
    			//(1)End if it is not (dummy or npu)*********************
    			//(2)else if it is dummy or npu *******************************************
    			else if(Layer!=Layers){
    #if NPU_Debug
    				std::cerr<<"Attach layer: finalizing Dummy or NPU subgraph\n";
    #endif
    				common_params.labels="";
    				(*sub_graph)<<OutputLayer(get_Sender_accessor(common_params, 0, 0));
    				//NPU:
    				if(gr_layer[Layer-1]==-2){
    #if NPU_Debug
    					std::cerr<<"Attach layer: subgraph is npu\n";
    #endif
    					common_params.labels="npu";
    					if(gr_layer[Layer]==-1)
    						common_params.labels="npu_to_dummy";
    					unsigned int output_size=1;
    					for(auto &node : sub_graph->graph().nodes())
    					{
    						if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
    						{
    							output_size=node->input(0)->desc().shape.total_size();
    						}
    					}
    #if NPU_Debug
    					std::cerr<<"Attach layer: found output size of subgraph\n";
    					std::cerr<<"Con_id: "<<Connection_id<<" and T_id: "<<T_id<<'\n';
    #endif

    					//NPU_Senders.push_back(dynamic_cast<std::unique_ptr<arm_compute::graph_utils::SenderAccessor>>(get_Sender_accessor(common_params,Connection_id)));
    					arm_compute::graph_utils::SenderAccessor* r=dynamic_cast<arm_compute::graph_utils::SenderAccessor*>(get_Sender_accessor(common_params, Connection_id, T_id, output_size,&NPU_Contexts[NPU_index]).release());

    					arm_compute::graph_utils::NPU_Senders.push_back(r);
    #if NPU_Debug
    					for(int i=0;i<NPU_Senders.size();i++){
    						std::cerr<<"att_1:\nNPU Send: "<<i<<" Con id: "<<NPU_Senders[i]->get_connection_id()<<'\n';
    					}
    					for(int i=0;i<NPU_Receivers.size();i++){
    						std::cerr<<"att_2:\nNPU Rec: "<<i<<" Con id: "<<NPU_Receivers[i]->get_connection_id()<<'\n';
    					}
    					std::cerr<<"Attach layer: Add sender to senders\n";
    #endif

    					/*npu_init_context(NPU_index);
    					std::cerr<<"Attach layer: init npu model\n";
    					if(NPU_index==0){
    						//Input_Accessor=get_input_accessor(common_params, std::move(preprocessor), true, NPU_Contexts[NPU_index]).get();
    						Input_Accessor=get_input_accessor(common_params, NULL, true, &NPU_Contexts[NPU_index]).get();
    						im_acc=dynamic_cast<ImageAccessor*>(Input_Accessor);
    					}
    					NPU_index++;*/

    				}
    			}
    			//(2)************************************************************************

    			//(3)if This is not last subgraph --> prepare next subgraph********************************
    			if(Layer!=Layers){
    #if NPU_Debug
    				std::cerr<<"Attach layer: preparing next subgraph\n";
    #endif
    				arm_compute::graph::Tensor* temp_sender;
    				TensorShape tshape;
    				//if(gr_layer[Layer]!=-1){
    				//set tensor shape for input of next subgraph
    				for(auto &node : sub_graph->graph().nodes())
    				{
    					if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
    					{
    						//Consider NPU
    						//if both subgraphs are real
    						//HERE:if(gr_layer[Layer-1]>-1 && gr_layer[Layer]!=-1){
    						if(gr_layer[Layer-1]>-1 && gr_layer[Layer]>-1){
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
    #if NPU_Debug
    				std::cerr<<"Attach layer: findout size of sender for previous subgraph\n";
    #endif

    				//(3-1) if next subgraph is not dummy or NPU
    				if(gr_layer[Layer]>-1){
    #if NPU_Debug
    					std::cerr<<"Attach layer: next subgraph is normal, preparing its input\n";
    #endif
    					sub_graph=(graphs[gr_layer[Layer]]);

    					if(gr_layer[Layer-1]==-1){
    						common_params.image="";
    					}
    					//NPU:
    					else if(gr_layer[Layer-1]==-2){
    						common_params.image="transfer_from_npu";
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
    #if NPU_Debug
    					std::cerr<<"\n\n**********\nShape:::"<<tshape<<"\n\n*******\n";
    #endif
    					//const TensorShape tensor_shape     = permute_shape(tshape, DataLayout::NCHW, operation_layout);
    					arm_compute::PermutationVector pv=arm_compute::PermutationVector(1U, 2U, 0U);
    					TensorShape tensor_shape=tshape;
    					arm_compute::permute(tensor_shape, pv);
    #if NPU_Debug
    					std::cerr<<"\n\n**********\nShape:::"<<tensor_shape<<"\n\n*******\n";
    #endif

    					//TensorDescriptor input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(DataLayout::NCHW);
    					TensorDescriptor input_descriptor = TensorDescriptor(tshape, common_params.data_type).set_layout(operation_layout);
    					(*sub_graph) << common_params.target
    								  << common_params.fast_math_hint;
    					//std::cout<<common_params.image<<", "<<Transmitters.size()-1<<std::endl;

    					//auto tt=InputLayer(input_descriptor, get_Receiver_accessor(common_params,gr_layer[Layer]-1));
    					//auto tt=InputLayer(input_descriptor, get_Receiver_accessor(common_params,Transmitters.size()-1));


    					//(*sub_graph)<<InputLayer(input_descriptor, get_Receiver_accessor(common_params,Transmitters.size()-1));
    #if NPU_Debug
    					std::cerr<<"Attach layer: next subgraph image:"<<common_params.image<<std::endl;
    #endif
    					if(gr_layer[Layer-1]==-2){
    #if NPU_Debug
    						std::cerr<<"Attach layer: Previous subgraph was npu\n";
    #endif
    						//arm_compute::graph_utils::SenderAccessor* T=dynamic_cast<arm_compute::graph_utils::SenderAccessor*>(Transmitters[Transmitters.size()-1]->accessor());
    						//(*sub_graph)<<InputLayer(input_descriptor, get_Receiver_accessor(common_params,Connection_id,0,NULL,T));

    						(*sub_graph)<<InputLayer(input_descriptor, get_Receiver_accessor(common_params, Connection_id, T_id, 0, NULL,NPU_Senders[NPU_Senders.size()-1], Transpose));
    #if NPU_Debug
    						std::cerr<<"Attach layer: added form npu\n";
    #endif
    					}
    					else{
    						(*sub_graph)<<InputLayer(input_descriptor, get_Receiver_accessor(common_params,Connection_id, T_id));
    					}

    					Connection_id++;
    					if(gr_layer[Layer-1]>-1){
    						T_id++;
    					}

    					cpu_set_t set;
    					CPU_ZERO(&set);
    					//CPU_SET(host_core[classes[gr_layer[Layer]]],&set);
    					set_cores(&set,one_master_core,host_core[classes[gr_layer[Layer]]]);
    					ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
    #if NPU_Debug
    					std::cerr<<"Attach layer: next normal subgraph is prepared\n";
    #endif
    				}
    				//(3-1)End if next subgraph is not dummy or npu***************
    				//(3-2) if next subgraph is dummy or npu***********
    				else{
    #if NPU_Debug
    					std::cerr<<"Attach layer: next subgraph is dummy or npu, preparing it\n";
    #endif
    					delete dump_graph;
    					dump_graph=new Stream(1000,"SqueezeNet");
    					sub_graph=dump_graph;
    					common_params.target=arm_compute::graph::Target ::NEON;
    					const auto        operation_layout = common_params.data_layout;
    					TensorDescriptor input_descriptor = TensorDescriptor(tshape, common_params.data_type).set_layout(operation_layout);
    					(*sub_graph) << common_params.target
    								  << common_params.fast_math_hint;
    					common_params.image="";
    					(*sub_graph)<<InputLayer(input_descriptor, get_Receiver_accessor(common_params,0,0));
    					//NPU:
    					if(gr_layer[Layer]==-2){
    #if NPU_Debug
    						std::cerr<<"Attach layer: next subgraph is npu, preparing it\n";
    #endif
    						NPU_index++;
    						npu_init_context(NPU_index);
    #if NPU_Debug
    						std::cerr<<"Attach layer: init npu model\n";
    #endif
    						/*if(NPU_index==0){
    							//Input_Accessor=get_input_accessor(common_params, std::move(preprocessor), true, NPU_Contexts[NPU_index]).get();
    							Input_Accessor=get_input_accessor(common_params, NULL, true, &NPU_Contexts[NPU_index]).get();
    							im_acc=dynamic_cast<ImageAccessor*>(Input_Accessor);
    						}*/

    						common_params.image="npu";
    						if(gr_layer[Layer-1]==-1){
    							common_params.image="npu_from_dummy";
    						}
    						unsigned int input_size=tshape.total_size();
    #if NPU_Debug
    						std::cerr<<"Attach layer: get total size of the input\n";
    #endif
    						arm_compute::graph_utils::ReceiverAccessor* r=dynamic_cast<arm_compute::graph_utils::ReceiverAccessor*>(get_Receiver_accessor(common_params, Connection_id, T_id, input_size, &NPU_Contexts[NPU_index]).release());
    #if NPU_Debug
    						std::cerr<<"Attach layer: get receiver of the npu subgarph\n";
    #endif
    						arm_compute::graph_utils::NPU_Receivers.push_back(r);
    #if NPU_Debug
    						std::cerr<<"bebin\ncon: "<<Connection_id<<" tedad: "<<NPU_Receivers.size()<<" unja: "<<NPU_Receivers[NPU_Receivers.size()-1]->get_connection_id()<<std::endl;
    						std::cerr<<"Attach layer: Add its Receiver into NPU Receivers\n";
    #endif
    						if(gr_layer[Layer-1]>-1){
    #if NPU_Debug
    							std::cerr<<"Attach layer: Setting NPU Receiver into previous normal subgraph\n";
    #endif
    							//arm_compute::graph_utils::SenderAccessor* T=dynamic_cast<arm_compute::graph_utils::SenderAccessor*>(Transmitters[Transmitters.size()-1]->accessor());
    							//T->set_receiver_accessor(r);
    							//HERE:dynamic_cast<arm_compute::graph_utils::SenderAccessor*>(Transmitters[Transmitters.size()-1]->accessor())->set_receiver_accessor(r);
    							dynamic_cast<arm_compute::graph_utils::SenderAccessor*>(temp_sender->accessor())->set_receiver_accessor(r);
    #if NPU_Debug
    							std::cerr<<"Attach layer: Set NPU Receiver into previous normal subgraph done\n";
    #endif
    						}

    						//std::unique_ptr<arm_compute::graph_utils::ReceiverAccessor> r(static_cast<arm_compute::graph_utils::ReceiverAccessor*>(get_Receiver_accessor(common_params,Connection_id).release()));
    						//arm_compute::graph_utils::NPU_Receivers.push_back(std::move(r));

    						Connection_id++;
    					}
    #if NPU_Debug
    					std::cerr<<"Attach layer: next dummy or npu subgrpah is prepared\n";
    #endif
    				}
    				//(3-2) End if next subgraph is dummy
    			}
    			//(3)End of if not last graph *********************************************************


    			start_Layer=Layer;
    		}
    		//(0)End of if subgraph finished ********************************************
    #if NPU_Debug
    		for(int i=0;i<NPU_Receivers.size();i++){
    			std::cerr<<"att:\nNPU Rec: "<<i<<" Con id: "<<NPU_Receivers[i]->get_connection_id()<<'\n';
    		}
    		std::cerr<<"*********************************Attached\n\n\n";
    #endif

        }

	void do_finish(){
		del();
		return;
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
        const TensorShape tensor_shape     = permute_shape(TensorShape(227U, 227U, 3U, 1U), DataLayout::NCHW, operation_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;


        //Ehsan
        //**********************************************************************************
        fill_host_core();
#if Power_Measurement
		if (-1 == GPIOExport(POUT))
				return(1);
		if (-1 == GPIODirection(POUT, OUT))
				return(2);

#endif

#if Power_Measurement
		//Power
		if (-1 == GPIOWrite(POUT, 0))
			std::cerr<<"Could not write 0 to GPIO\n";
#endif
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
        //NPU
		int start_N=-1;
		int end_N=-1;
		for(int i=0;i<Layers;i++){
			if (order[i]=='N'){
				if (start_N==-1){
					start_N=i;
					end_N=i;
				}
				else{
					end_N=i;
				}
			}
		}
		NPU_Model_Name=NPU_Model_Name+'_'+std::to_string(start_N+1)+'_'+std::to_string(end_N+1)+".rknn";
		for(int i=0;i<Layers;i++){
			if(i==0){
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
				if (order[i]=='N'){
					//targets.push_back(arm_compute::graph::Target ::NEON);
					//classes.push_back(3);
					rknn_context ctx=0;
					NPU_Contexts.push_back(ctx);
				}
				if (order[i]!='-' && order[i]!='N' ){
					graphs.push_back(new Stream(g+1,"SqueezeNet"));
					gr_layer[i]=g;
				}
				if(order[i]=='-'){
					gr_layer[i]=-1;
				}
				if(order[i]=='N'){
					gr_layer[i]=-2;
				}
			}

			else if (order[i]!=order[i-1]){
				if(order[i]=='-'){
					gr_layer[i]=-1;
				}
				else if(order[i]=='N'){
					gr_layer[i]=-2;
					rknn_context ctx=0;
					NPU_Contexts.push_back(ctx);
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

					graphs.push_back(new Stream(g,"SqueezeNet"));
					gr_layer[i]=graphs.size()-1;
					g=graphs.size()-1;
				}

			}

			else{
				if(order[i]!='-' && order[i]!='N')
					gr_layer[i]=g;
				if(order[i]=='-')
					gr_layer[i]=-1;
				if(order[i]=='N')
					gr_layer[i]=-2;
			}
		}
		for(int i=0;i<Layers;i++){
			//std::cerr<<i<<"\t"<<gr_layer[i]<<std::endl;
			if(order[i]=='-' || order[i]=='N'){
				dump_graph=new Stream(1000,"SqueezeNEt");
				break;
			}
		}
#if NPU_Debug
		std::cerr<<"graph layers:\n";
		for(int i=0;i<Layers;i++){
					std::cerr<<i<<"\t"<<gr_layer[i]<<std::endl;
		}
#endif
		////per_frame=(graphs.size()>1);
		/*for(int i=0;i<8;i++){
			std::cout<<"Layer:"<<i<<'\t'<<"graph:"<<gr_layer[i]<<'\t'<<"class:"<<classes[gr_layer[i]]<<'\t'<<"target:"<<int(targets[gr_layer[i]])<<std::endl;
		}*/


		cpu_set_t set;
		CPU_ZERO(&set);
		//NPU:
		//
		if(gr_layer[Layer]>0){
			//CPU_SET(host_core[classes[gr_layer[Layer]]],&set);
			set_cores(&set,one_master_core,host_core[classes[gr_layer[Layer]]]);
			ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
		}
		std::cout << common_params << std::endl;

		annotate=common_params.annotate;
		//ann=annotate;
		save_model=common_params.save;

		//If subgraph is dummy
		if(gr_layer[Layer]==-1 ){
			sub_graph=dump_graph;
			common_params.target=arm_compute::graph::Target ::NEON;
		}

		//If subgraph is NPU
		else if(gr_layer[Layer]==-2){
			sub_graph=dump_graph;
			common_params.target=arm_compute::graph::Target ::NEON;
			NPU_index++;
			npu_init_context(NPU_index);
#if NPU_Debug
			std::cerr<<"Setup: init npu model\n";
#endif
			//Input_Accessor=get_input_accessor(common_params, std::move(preprocessor), true, NPU_Contexts[NPU_index]).get();
			Input_Accessor=get_input_accessor(common_params,std::move(preprocessor), true, &NPU_Contexts[NPU_index],tensor_shape.total_size()).release();
			im_acc=dynamic_cast<ImageAccessor*>(Input_Accessor);

			arm_compute::TensorInfo info(input_descriptor.shape,1,input_descriptor.data_type,operation_layout);
			Input_tensor.allocator()->init(info);
			Input_tensor.allocator()->allocate();

		}
		//If subgraph is real
		else{
			sub_graph=(graphs[gr_layer[Layer]]);
			common_params.target=targets[gr_layer[Layer]];
		}
		//***************************************************************




		std::cerr<<"start creating graph\n";
        (*sub_graph) << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, (gr_layer[Layer]==-2)?NULL:std::move(preprocessor)));
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

		//im_acc=dynamic_cast<ImageAccessor*>(graphs[0]->graph().node(0)->output(0)->accessor());
		if(gr_layer[Layer-1]==-2){
			Output_Accessor=get_output_accessor(common_params, 5, false, std::cout, &NPU_Contexts[NPU_index]).release();
		}
		if(gr_layer[0]!=-2)
			im_acc=dynamic_cast<ImageAccessor*>(graphs[0]->graph().node(0)->output(0)->accessor());

		for(int i=0;i<NPU_Contexts.size();i++){
			NPU_time.push_back(0.0);
		}

		std::cout<<"Total layers:"<<Layer<<std::endl<<std::endl;

#if NPU_Debug
		for(int i=0;i<NPU_Receivers.size();i++){
			std::cerr<<"setup:\nNPU Rec: "<<i<<" Con id: "<<NPU_Receivers[i]->get_connection_id()<<'\n';
		}
#endif




		return true;
    }






    void do_run() override
    {
        // Run graph
        //Ehsan
    	cpu_set_t set;
    	CPU_ZERO(&set);
    	CPU_SET(1,&set);
    	ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
    	//std::string t;
#if NPU_Debug
    	std::cerr<<"\n\n\n_________________________\n\n\nSize of NPU contexts: "<<NPU_Contexts.size()
    			<<"\nSize of NPU senders: "<<NPU_Senders.size()
				<<"\nSize of NPU receivers: "<<NPU_Receivers.size()
				<<"\nNumber of ARMCL subgraphs: "<<graphs.size()
				<<"\nNumber of NPU subgraphs: "<<NPU_Contexts.size()
				<<std::endl;
#endif
#if NPU_Debug
    	for(int i=0;i<NPU_Senders.size();i++){
			std::cerr<<"att_1:\nNPU Send: "<<i<<" Con id: "<<NPU_Senders[i]->get_connection_id()<<'\n';
		}
#endif
    	std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    	std::vector<std::thread*> stages;
    	std::vector<std::thread*> npu_stages;
    	int n=common_params.n;
    	//NPU:
		for(int i=0;i<NPU_Contexts.size();i++){
			npu_stages.push_back(new std::thread(&GraphSqueezenetExample::run_npu,this,i));
		}
		//std::this_thread::sleep_for(std::chrono::milliseconds(4000));
    	for(int i=0;i<graphs.size();i++){
    		//std::cerr<<"creating thread "<< i<<"\n";
    		stages.push_back(new std::thread(&GraphSqueezenetExample::run,this,i));
    		//std::cerr<<"thread "<< i<<" created\n";
    		//stages[i]->join();
    	}


    	for(int i=0;i<stages.size();i++){
			stages[i]->join();
    	}
    	for(int i=0;i<npu_stages.size();i++){
			npu_stages[i]->join();
		}
    	for(int i=0;i<graphs.size();i++){
			//std::cout<<"graph_id: "<<i<<" \t start: "<<graphs[i]->get_start_time().time_since_epoch().count()<<" \t end: "<<graphs[i]->get_finish_time().time_since_epoch().count()<<std::endl;
    		if(common_params.layer_time)
    				graphs[i]->measure(n);

			double tot=graphs[i]->get_input_time()+graphs[i]->get_task_time()+graphs[i]->get_output_time();
			PrintThread{}<<"\n\nCost"<<i<<":"<<1000*graphs[i]->get_cost_time()/n<<std::endl;
			PrintThread{}<<"input"<<i<<"_time:"<<1000*graphs[i]->get_input_time()/n<<"\ntask"<<i<<"_time:"<<1000*graphs[i]->get_task_time()/n<<"\noutput"<<i<<"_time:"<<1000*graphs[i]->get_output_time()/n<<"\ntotal"<<i<<"_time:"<<1000*tot/n<<std::endl;
			PrintThread{}<<"***************************************\n\n";

		}
    	for(int i=0;i<NPU_time.size();i++){
    		PrintThread{}<<"\nNPU subgraph: "<<i<<" --> Cost: "<<NPU_time[i]*1000/n<<"\n\n";
    		PrintThread{}<<"******************************************************\n\n";

    	}


    	PrintThread{}<<"Frame Latency: "<<1000*latency<<std::endl;



    	//Power
    	/*if (-1 == GPIOWrite(POUT, 1))
    			std::cerr<<"could not write 1\n";*/
#if Power_Measurement
    	if (-1 == GPIOUnexport(POUT))
    			std::cerr<<"could not unexport\n";
#endif
    	//del();

    }
    void run(int graph_id){
    	//std::cerr<<"setup finished now start running\n";
		int cl=classes[graph_id];
		int core_id=host_core[cl];
		cpu_set_t set;
		CPU_ZERO(&set);
		//CPU_SET(core_id,&set);
		set_cores(&set,one_master_core,core_id);
		ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
		//PrintThread{}<<"start running graph "<<graph_id<<std::flush<<std::endl;

		double in=0;
		double task=0;
		double out=0;
		int n=(common_params.n);
		bool layer_timing=common_params.layer_time;
		bool ending=(graph_id==graphs.size()-1)&&(Output_Accessor==NULL);
		bool starting=(graph_id==0)&&(Input_Accessor==NULL);
		latency=0;
		//auto tstart=std::chrono::high_resolution_clock::now();
		//std::cerr<<"graph__id:"<<graph_id<<'\n';//"   time:"<<tstart.time_since_epoch().count()<<std::endl;
		if(imgs && starting){
			if(image_index>=images_list.size())
					image_index=image_index%images_list.size();
			PrintThread{}<<"\n\nFirst graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
			//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
			im_acc->set_filename(images_list[image_index]);
#if NPU_Debug
			PrintThread{}<<"\n\nSet file name done!\n"<<std::endl;
#endif
		}
		if(layer_timing){
			//std::cerr<<i<<" graph_id:"<<graph_id<<"   time:"<<std::chrono::high_resolution_clock::now().time_since_epoch().count()<<std::endl;
			graphs[graph_id]->run(annotate);
			//graphs[graph_id]->set_finish_time(std::chrono::high_resolution_clock::now());
		}
		else{
			graphs[graph_id]->run(annotate);
		}
		//std::cerr<<"run: finish running graph id: "<<graph_id<<'\n';

		graphs[graph_id]->set_input_time(0);
		graphs[graph_id]->set_task_time(0);
		graphs[graph_id]->set_output_time(0);
		graphs[graph_id]->set_cost_time(0);

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		PrintThread{}<<"\nrun: Start running graph "<<graph_id<<std::flush<<std::endl;
		/*if(starting){
			std::this_thread::sleep_for(std::chrono::milliseconds(2000));
		}*/
		if(layer_timing)
			graphs[graph_id]->reset();
		auto tstart=std::chrono::high_resolution_clock::now();

		//std::cout<<tstart.time_since_epoch().count()<<std::endl;

#if Power_Measurement
		//Power
		if (-1 == GPIOWrite(POUT, 1))
			std::cerr<<"Could not write to GPIO\n";
#endif

		int iii=n/2;
		for(int i=0;i<n;i++){
			if(starting && i==iii){
				std::cerr<<"start of graph: "<<graph_id<<" for frame: "<<i<<std::endl;
				start=std::chrono::high_resolution_clock::now();
			}
			if(imgs && starting){
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

			}
			else{
				graphs[graph_id]->run(annotate);
			}
			if(ending && i==iii){
				//std::cerr<<"end of graph: "<<graph_id<<" for frame: "<<i<<std::endl;
				//latency += std::chrono::duration_cast<std::chrono::duration<double>>(graphs[graph_id]->get_finish_time() - graphs[0]->get_start_time()).count();
				latency += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
				//start=std::chrono::high_resolution_clock::now();
			}
		}
#if Power_Measurement
		if (-1 == GPIOWrite(POUT, 0))
		    std::cerr<<"could not write 1\n";
#endif
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

    //Run NPU
    void run_npu(int id){
		//int cl=classes[graph_id];
    	//std::cerr<<"Run NPU\n";
    	//int core_id=common_params.total_cores-(id+1);
		cpu_set_t set;
		CPU_ZERO(&set);
		//CPU_SET(core_id,&set);
		////set_cores(&set,one_master_core,core_id);
		set_cores(&set,one_master_core,npu_host);
		ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
		//PrintThread{}<<"start running graph "<<graph_id<<std::flush<<std::endl;
		std::cerr<<"\nnpu_run: Start running NPU "<<id<<std::flush<<std::endl;
		double in=0;
		double task=0;
		double out=0;
		int n=(common_params.n);
		//bool layer_timing=common_params.layer_time;
		//bool end=(id==NPU_Contexts.size()-1);
		bool starting=(Input_Accessor!=NULL);
		bool ending=(Output_Accessor!=NULL);
		latency=0;
		//auto tstart=std::chrono::high_resolution_clock::now();
		//std::cerr<<"graph__id:"<<graph_id<<"   time:"<<tstart.time_since_epoch().count()<<std::endl;
		//if(imgs && id==0){
		if(starting && imgs){
			if(image_index>=images_list.size())
					image_index=image_index%images_list.size();
			PrintThread{}<<"\n\nFirst graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
			//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
			im_acc->set_filename(images_list[image_index]);
		}
		/*if(layer_timing){
			//std::cerr<<i<<" graph_id:"<<graph_id<<"   time:"<<std::chrono::high_resolution_clock::now().time_since_epoch().count()<<std::endl;
			graphs[graph_id]->run(annotate);
			//graphs[graph_id]->set_finish_time(std::chrono::high_resolution_clock::now());
		}
		else{
			graphs[graph_id]->run(annotate);
		}*/
		if(starting){
#if NPU_Debug
			std::cerr<<"npu_run: Calling NPU Input accessor id: "<<id<<'\n';
#endif
			Input_Accessor->access_tensor(Input_tensor);
#if NPU_Debug
			std::cerr<<"npu_run: Finish Calling NPU Input accessor id: "<<id<<'\n';
#endif
		}
		else{
#if NPU_Debug
			std::cerr<<"npu_run: Calling NPU Receiver id: "<<id<<'\n';
#endif
			NPU_Receivers[id]->access_tensor(*temp_tensor);
#if NPU_Debug
			std::cerr<<"npu_run: Finish Calling NPU Receiver id: "<<id<<'\n';
#endif
		}
#if NPU_Debug
		std::cerr<<"npu_run: Calling NPU Run id: "<<id<<'\n';
#endif
		int ret = rknn_run(NPU_Contexts[id], NULL);
		if(ret<0){
			std::cerr<<"npu_run: Error "<<ret<<" running NPU part with id: "<<id<<std::endl;
		}
#if NPU_Debug
		std::cerr<<"npu_run: Finish Calling NPU Run id: "<<id<<'\n';


		for(int i=0;i<NPU_Senders.size();i++){
			std::cerr<<"att_1:\nNPU Send: "<<i<<" Con id: "<<NPU_Senders[i]->get_connection_id()<<'\n';
		}
#endif
		if(ending){
#if NPU_Debug
			std::cerr<<"npu_run: Calling NPU Output id: "<<id<<'\n';
#endif
			/*if(&(*temp_tensor)==NULL){
				std::cerr<<"temp is null\n";
			}*/
			Output_Accessor->access_tensor(*temp_tensor);
		}
		else{
#if NPU_Debug
			std::cerr<<"npu_run: Calling NPU Sender id: "<<id<<'\n';
#endif
			//ITensor *temp_tensor2;
			NPU_Senders[id]->access_tensor(*temp_tensor);
#if NPU_Debug
			std::cerr<<"npu_run: Finish Calling NPU Sender id: "<<id<<'\n';
#endif
		}

		/*graphs[graph_id]->set_input_time(0);
		graphs[graph_id]->set_task_time(0);
		graphs[graph_id]->set_output_time(0);
		graphs[graph_id]->set_cost_time(0);
		if(layer_timing)
			graphs[graph_id]->reset();*/
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		//if(id==0){
		/*if(starting){
			std::this_thread::sleep_for(std::chrono::milliseconds(2000));
		}*/
		auto tstart=std::chrono::high_resolution_clock::now();

		//std::cout<<tstart.time_since_epoch().count()<<std::endl;

#if Power_Measurement
		//Power
		if (-1 == GPIOWrite(POUT, 1))
			std::cerr<<"Could not write to GPIO\n";
#endif
		int iii=n/2;
		for(int i=0;i<n;i++){
			//if(id==0){
			if(starting){
#if NPU_Debug
				std::cerr<<"Input accessor is not null!\n";
#endif
				if(i==iii)
					start=std::chrono::high_resolution_clock::now();
				if(imgs){
					if(image_index>=images_list.size())
							image_index=image_index%images_list.size();
					PrintThread{}<<"\n\nFirst graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
					//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
					im_acc->set_filename(images_list[image_index++]);
#if NPU_Debug
					std::cerr<<"npu_run: Calling NPU Image accessor id: "<<id<<'\n';
#endif
					im_acc->access_tensor(Input_tensor);
#if NPU_Debug
					std::cerr<<"npu_run: Finish Calling NPU Image accessor id: "<<id<<'\n';
#endif
				}
				else{
#if NPU_Debug
					std::cerr<<"npu_run: Calling NPU Input accessor id: "<<id<<'\n';
#endif
					Input_Accessor->access_tensor(*temp_tensor);
#if NPU_Debug
					std::cerr<<"npu_run: Finish Calling NPU Input accessor id: "<<id<<'\n';
#endif
				}
			}
			else{
#if NPU_Debug
				std::cerr<<"npu_run: Calling NPU Rec accessor id: "<<id<<'\n';
#endif
				NPU_Receivers[id]->access_tensor(*temp_tensor);
#if NPU_Debug
				std::cerr<<"npu_run: Finsih Calling NPU Rec accessor id: "<<id<<'\n';
#endif
			}
#if NPU_Debug
			std::cerr<<"npu_run: Calling NPU Run id: "<<id<<'\n';
#endif
			ret = rknn_run(NPU_Contexts[id], NULL);
#if NPU_Debug
			std::cerr<<"npu_run: Finish Calling NPU Run id: "<<id<<'\n';
#endif
			if(ret<0){
				//std::string c;
				std::cerr<<"Error "<<ret<<" running NPU part with id: "<<id<<std::endl;
				//std::cin>>c;
			}

			if(ending){
#if NPU_Debug
				std::cerr<<"npu_run: Calling NPU Output accessor id: "<<id<<'\n';
#endif
				Output_Accessor->access_tensor(*temp_tensor);
				if(i==iii)
					latency += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
#if NPU_Debug
				std::cerr<<"npu_run: Finish Calling NPU Output accessor id: "<<id<<'\n';

				start=std::chrono::high_resolution_clock::now();
#endif
			}
			else{
#if NPU_Debug
				std::cerr<<"npu_run: Calling NPU Sender accessor id: "<<id<<'\n';
#endif
				NPU_Senders[id]->access_tensor(*temp_tensor);
#if NPU_Debug
				std::cerr<<"npu_run: Finish Calling NPU Sender accessor id: "<<id<<'\n';
#endif
			}
			/*if(layer_timing){
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
			}*/
		}
#if Power_Measurement
		if (-1 == GPIOWrite(POUT, 0))
		    std::cerr<<"could not write 1\n";
#endif
		auto tfinish=std::chrono::high_resolution_clock::now();
		double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		NPU_time[id]=cost0;
		//graphs[graph_id]->set_input_time(in);
		//graphs[graph_id]->set_task_time(task);
		//graphs[graph_id]->set_output_time(out);
		////graphs[graph_id]->set_cost_time(cost0);
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
    bool one_master_core=false;
    std::map<int, int> host_core = {{0, 1}, {1, 5}, {2, 4}};
    ImageAccessor *im_acc=NULL;
    Stream *dump_graph=NULL;
    std::map<int,int> gr_layer;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    //std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    double latency=0;
    //NPU:
    std::vector<rknn_context> NPU_Contexts;
    std::vector<double> NPU_time;
    std::string NPU_Model_Name="Squeeze";
    int NPU_index=-1;
    arm_compute::graph::ITensorAccessor *Output_Accessor=NULL;
    arm_compute::graph::ITensorAccessor *Input_Accessor=NULL;
    //std::map<int, int> core_npu = {{},{},{}};
    bool Transpose=true;

    int npu_host;

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
