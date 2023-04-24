/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#define streamline 0
#include<chrono>
#ifndef My_print
#include "arm_compute/gl_vs.h"
#endif
#if streamline > 0
#include"annotate/Sr_ann.c"
#endif
//#include "src/graph/GraphManager.cpp"

#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManager.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/BackendRegistry.h"

//For AOA
#include <fstream>
#include <unordered_map>
//#include <algorithm> // Include the <algorithm> header for std::replace()


//#include "power.h"
#include "utils/Power.h"

namespace arm_compute
{
namespace graph
{
namespace detail
{

#define PROFILE_MODE_LAYERS 1
#define PROFILE_MODE_WHOLE_NETWORK 2
#define PROFILE_MODE_TRANSFER_TIMES 3
#define PROFILE_MODE_SYNTHETIC_TRANSFERS 4
#define AOA 5

//#define PROFILE_MODE PROFILE_MODE_WHOLE_NETWORK
#define PROFILE_MODE AOA



/*
 * AOA algorithm; baseline for PELSI work
 */
#if PROFILE_MODE == AOA

std::chrono::time_point<std::chrono::high_resolution_clock> GlobalStartTime;
int LayerNumber = 0;
double TargetLatency=250; //ms
double Gamma=0.2;
int LFreqInit=0;
int BFreqInit=0;
int GFreqInit=0;
int CurLittleFreq,CurBigFreq,CurGPUFreq;
double elapsed_Task_Percent=0;


/*
unsigned long long total_time_diff = 0;
unsigned long long prev_total_time = 0;
unsigned long long curr_total_time_small = 0;
unsigned long long prev_idle_time = 0;
unsigned long long curr_idle_time = 0;
*/
// Function to get CPU usage statistics
double getCpuStats() {
	static unsigned long long prev_total_time = 0;
	static unsigned long long prev_idle_time = 0;
    std::ifstream file("/proc/stat");
    std::string line;
    //unsigned long long user, nice, system, idle;
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
    unsigned long long total_time = 0;
    unsigned long long idle_time = 0;
    total_time = 0;
    idle_time = 0;

    if (file.is_open()) {
        std::getline(file, line);
        /*
        sscanf(line.c_str(), "cpu %llu %llu %llu %llu", &user, &nice, &system, &idle);
        total_time = user + nice + system + idle;
        idle_time = idle;
        */
        sscanf(line.c_str(), "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
           &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal, &guest, &guest_nice);
        total_time = user + nice + system + idle + iowait + irq + softirq + steal;
        idle_time = idle + iowait;
    }
    file.close();
    unsigned long long diff_idle=idle_time-prev_idle_time;
    unsigned long long diff_total=total_time-prev_total_time;
    double util=100*(1- (diff_idle/diff_total) );
    prev_total_time=total_time;
    prev_idle_time=idle_time;
    return util;
}

std::string active="/sys/devices/platform/ff9a0000.gpu/power/runtime_active_time";
std::string suspend="/sys/devices/platform/ff9a0000.gpu/power/runtime_suspended_time";

// Function to get CPU usage statistics
double getGpuStats( ) {
	static unsigned long long prev_active_time=0;
	static unsigned long long prev_idle_time=0;
	unsigned long long active_time;
	unsigned long long idle_time;
    std::ifstream file_active(active.c_str());
    std::ifstream file_suspend(suspend.c_str());
    std::string line;
    if (file_active.is_open()) {
        std::getline(file_active, line);
        sscanf(line.c_str(), "%llu", &active_time);
        //std::cerr<<"active: "<<active_time<<std::endl;
    }
    file_active.close();

    if (file_suspend.is_open()) {
        std::getline(file_suspend, line);
        sscanf(line.c_str(), "%llu", &idle_time);
        //std::cerr<<"idle: "<<idle_time<<std::endl;
    }
    file_suspend.close();
    unsigned long long diff_active=active_time-prev_active_time;
    unsigned long long diff_idle=idle_time-prev_idle_time;
    prev_idle_time=idle_time;
    prev_active_time=active_time;
    double u=100*(diff_active/(diff_active+diff_idle));
    return u;
}

//For loading the Layer Percentage
struct Key {
    std::string graph;
    int layer;
    std::string component;
};

bool operator==(const Key& lhs, const Key& rhs) {
    return lhs.graph == rhs.graph && lhs.layer == rhs.layer && lhs.component == rhs.component;
}

// Hash function for Key to be used in unordered_map
struct KeyHash {
    std::size_t operator()(const Key& key) const {
        std::size_t h1 = std::hash<std::string>{}(key.graph);
        std::size_t h2 = std::hash<int>{}(key.layer);
        std::size_t h3 = std::hash<std::string>{}(key.component);
        return h1 ^ h2 ^ h3;
    }
};

std::unordered_map<Key, double, KeyHash> data;
bool Data_Loaded=false;

int Load_Layers_Percetage(){
	if(Data_Loaded){
		return 0;
	}

    std::ifstream file("Layers_Percentage.csv"); // Replace "data.csv" with your actual CSV file name
    if (!file.is_open()) {
        std::cout << "Failed to open file" << std::endl;
        return 1;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        std::string graph;
        int layer;
        double timePercentageB;
        double timePercentageG;
        double timePercentageL;
        double timePercentageAverage;

        if (iss >> graph >> layer >> timePercentageB >> timePercentageG >> timePercentageL >> timePercentageAverage) {
            // Create a key based on graph, layer, and component
            Key key{graph, layer, "B"};
            data[key] = timePercentageB;
            key.component = "G";
            data[key] = timePercentageG;
            key.component = "L";
            data[key] = timePercentageL;
            key.component = "Average";
            data[key] = timePercentageAverage;
        } else {
            std::cout << "Failed to parse line: " << line << std::endl;
        }
    }
    Data_Loaded=true;
    return 0;
}


#endif


void validate_all_nodes(Graph &g)
{
    auto &nodes = g.nodes();

    // Create tasks
    for(auto &node : nodes)
    {
        if(node != nullptr)Load_Layers_Percetage();
        {
            Target                    assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            Status                    status          = backend.validate_node(*node);
            ARM_COMPUTE_ERROR_ON_MSG(!bool(status), status.error_description().c_str());
        }
    }
}

void configure_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && tensor->handle() == nullptr)
        {
            Target                         target  = tensor->desc().target;
            backends::IDeviceBackend      &backend = backends::BackendRegistry::get().get_backend(target);
            std::unique_ptr<ITensorHandle> handle  = backend.create_tensor(*tensor);
            ARM_COMPUTE_ERROR_ON_MSG(!handle, "Couldn't create backend handle!");
            tensor->set_handle(std::move(handle));
        }
    }
}

void allocate_all_input_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        Tensor *tensor = node.input(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
#if My_print > 0
            //Ehsan
            std::cout<<"\nExecutionHelpers, Allocating input tensor for outpu node, node shape:"<<tensor->handle()->tensor().info()->tensor_shape()
            		<<" tensor shape:"<<tensor->desc().shape
					<<std::endl;
#endif
            tensor->handle()->allocate();
        }
    }
}

void allocate_all_output_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_outputs(); ++i)
    {
        Tensor *tensor = node.output(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configuLoad_Layers_Percetage();red!");
#if My_print > 0
            //Ehsan
            std::cout<<"\nExecutionHelpers, Allocating output tensor for input and const node, CLTensor shape:"<<tensor->handle()->tensor().info()->tensor_shape()
            		<<" tensor shape:"<<tensor->desc().shape
					<<std::endl;
#endif
            tensor->handle()->allocate();
        }
    }
}

void allocate_const_tensors(Graph &g)
{
    for(auto &node : g.nodes())
    {
        if(node != nullptr)
        {
            switch(node->type())
            {
                case NodeType::Const:
                case NodeType::Input:
                    allocate_all_output_tensors(*node);
                    break;
                case NodeType::Output:
                    allocate_all_input_tensors(*node);
                default:
                    break;
            }
        }
    }
}

void allocate_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && !tensor->bound_edges().empty() && tensor->handle() != nullptr && tensor->handle()->tensor().info()->is_resizable() && tensor->handle()->tensor().is_used())
        {
            tensor->handle()->allocate();
        }
    }
}

ExecutionWorkload configure_all_nodes(Graph &g, GraphContext &ctx, const std::vector<NodeID> &node_order)
{
#if PROFILE_MODE == AOA
	Load_Layers_Percetage();
#endif

    ExecutionWorkload workload;
    workload.graph = &g;
    workload.ctx   = &ctx;

    // Reserve memory for tasks
    workload.tasks.reserve(node_order.size());

    // Create tasks
    for(auto &node_id : node_order)
    {
        auto node = g.node(node_id);
        //Ehsan
        /*
        std::cout<<"\n*******************************\nnode name: "<<node->name()<<" ID: "<<node->id()<<" num inputs: "<<node->num_inputs()<<std::endl<<std::flush;
        for(int k=0; k < node->num_inputs(); k++){
        	INode *cc=node->input_edge(k)->producer();
        	std::cout<<"\ninput "<<k<<" node_name: "<<cc->name()<<" ID: "<<cc->id()<<std::endl<<std::flush;
        	TensorShape shape=node->input(k)->desc().shape;
        	std::cout<<shape<<std::endl;
            //for(int i=0;i<shape.num_dimensions();i++) std::cout<<shape[i]<<'\t'<<std::flush;
            //std::cout<<"Padding: "<<_padding.left<<_padding.right<<_padding.top<<_padding.bottom<<std::endl;
        }*/

        /*
         ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << CLTargetInfo::TargetType
                               << " Data Type: " << input0->info()->data_type()
                               << " Input0 shape: " << input0->info()->tensor_shape()
                               << " Input1 shape: " << input1->info()->tensor_shape()
                               << " Input2 shape: " << input2->info()->tensor_shape()
                               << " Output0 shape: " << output0->info()->tensor_shape()
                               << " Output1 shape: " << output1->info()->tensor_shape()
                               << " Output2 shape: " << output2->info()->tensor_shape()
                               << " Output3 shape: " << output3->info()->tensor_shape()
                               << " DetectionPostProcessLayer info: " << detect_info
                               << std::endl);
         */


        if(node != nullptr)
        {
            Target                     assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            std::unique_ptr<IFunction> func            = backend.configure_node(*node, ctx);
            if(func != nullptr || is_utility_node(node))
            {
                workload.tasks.emplace_back(ExecutionTask(std::move(func), node));
            }
        }
    }

    // Add inputs and outputs
    for(auto &node : g.nodes())
    {
        if(node != nullptr && node->type() == NodeType::Input)
        {
        	//Ehsan
        	//std::cout<<"\ninput node name and ID: "<<node->name()<<'_'<<node->id()<<std::endl;

            workload.inputs.push_back(node->output(0));
        }

        if(node != nullptr && node->type() == NodeType::Output)
        {
            workload.outputs.push_back(node->input(0));
            //Ehsan
            //std::cout<<"\noutput node name and ID: "<<node->name()<<'_'<<node->id()<<std::endl;

            continue;
        }
    }

    return workload;
}

void release_unused_tensors(Graph &g)
{
    for(auto &tensor : g.tensors())
    {
        if(tensor != nullptr && tensor->handle() != nullptr)
        {
            tensor->handle()->release_if_unused();
        }
    }
}

void call_tensor_accessor(Tensor *tensor)
{
    ARM_COMPUTE_ERROR_ON(!tensor);
    tensor->call_accessor();
}

void call_all_const_node_accessors(Graph &g)
{
    auto &nodes = g.nodes();

    for(auto &node : nodes)
    {
        if(node != nullptr && node->type() == NodeType::Const && node->num_outputs())
        {
            if(!node->output(0)->bound_edges().empty())
            {
#if My_print > 0
            	//Ehsan
            	std::cout<<"ExecutionHelpers, call all const node tensor accessors, node name and ID: "<<node->name()<<'_'<<node->id()<<std::endl;
#endif
            	call_tensor_accessor(node->output(0));
            }
        }
    }
}

bool call_all_input_node_accessors(ExecutionWorkload &workload)
{
    bool is_valid = true;
    //Ehsan: size of inputs is 1
    //std::string c;
    //std::cerr<<"inputs size: "<<workload.inputs.size()<<std::endl;
    //std::string t;
    //std::cin>>t;
    std::for_each(std::begin(workload.inputs), std::end(workload.inputs), [&](Tensor * input_tensor)
    {
#if My_print > 0
    	std::cerr<<"input accessorrr"<<std::endl;
    	std::cerr<<input_tensor->desc().shape <<std::endl;
#endif
        bool valid_input = (input_tensor != nullptr) && input_tensor->my_call_accessor();

#if PROFILE_MODE == AOA
      if (workload.graph->id()==0) {
    	  GlobalStartTime=std::chrono::high_resolution_clock::now();
    	  LayerNumber=0;
    	  elapsed_Task_Percent=0.0;
    	  //Just for setting prev_times inside this functions
    	  getCpuStats();
    	  getGpuStats();
      }

#endif
        is_valid         = is_valid && valid_input;
    });
    return is_valid;
}

void prepare_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.graph == nullptr);
    for(auto &task : workload.tasks)
    {
        task.prepare();
        release_unused_tensors(*workload.graph);
    }
}



void call_all_tasks(ExecutionWorkload &workload,int nn,bool last_graph)
{
    ARM_COMPUTE_ERROR_ON(workload.ctx == nullptr);

    // Acquire memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->acquire();
        }
    }


#if streamline > 0
    ANNOTATE_SETUP;
    ANNOTATE_MARKER_STR("start_running tasks");
    static int cc=0;
    static int c=0;
#endif
    std::string last_task_name=workload.tasks[workload.tasks.size()-1].node->name();
    for(auto &task : workload.tasks)
    {
    	if(nn==0)
    		task();
    	else{
#if streamline > 0
    		ANNOTATE_CHANNEL_COLOR(cc,((c%2)==0)?ANNOTATE_GREEN:ANNOTATE_YELLOW, (std::to_string(c)+" "+task.node->name()).c_str() );
#endif

    		if(task.starting){
    			if (-1 == GPIOWrite(POUT, 0)){
    				std::cerr<<"Could not write to GPIO\n";
    			}
			}



			bool last_layer=last_graph && task.node->name()==last_task_name;
#if PROFILE_MODE == PROFILE_MODE_LAYERS
			/************Profile layers time and power*********/
			task(nn);
			if(task.ending && !last_layer){
				task.apply_freq(task.node->name());
				if (-1 == GPIOWrite(POUT, 1)){
					std::cerr<<"Could not write to GPIO\n";
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(8));
			}
			/**************************************************/
#elif PROFILE_MODE == PROFILE_MODE_WHOLE_NETWORK
			/************Profiling whole network *********/
			task(nn);
			if(task.ending && !last_layer){
				task.apply_freq(task.node->name());
			}
			/**************************************************/
#elif PROFILE_MODE == PROFILE_MODE_TRANSFER_TIMES
			/************Profile transfer time between real layers *********/
			task(nn);
			if(task.ending && !last_layer){
				task.apply_freq(task.node->name());
				if (-1 == GPIOWrite(POUT, 1)){
					std::cerr<<"Could not write to GPIO\n";
				}
			}
			/***************************************************************/
#elif PROFILE_MODE == PROFILE_MODE_SYNTHETIC_TRANSFERS
			/***********Profiling synthetic transfers*********/
			if(task.ending){
				task.apply_freq(task.node->name());
				std::this_thread::sleep_for(std::chrono::milliseconds(15));
				if (-1 == GPIOWrite(POUT, 1)){
					std::cerr<<"Could not write to GPIO\n";
				}
			}
			/*************************************************/
#endif


#if PROFILE_MODE == AOA
			/************Profiling whole network *********/
			task(nn);
			if(task.ending && !last_layer){


				double U_CPU = getCpuStats();
				double U_GPU = getGpuStats();

				//Set based on target Latency and balancing parameter
			    std::string graphToFind = "alex";
			    int layerToFind = LayerNumber;
			    std::string componentToFind = "G";
			    Key keyToFind{graphToFind, layerToFind, componentToFind};
			    auto it = data.find(keyToFind);
			    double taskPercentage=0;
			    if (it != data.end()) {
			        taskPercentage = it->second;
			        std::cerr << "Time Percentage for " << graphToFind << " in component " << componentToFind
			                  << " for layer " << layerToFind << ": " << taskPercentage << std::endl;


			    } else {
			        std::cerr << "Data not found for the given key." << std::endl;
			    }

			    auto CurTime = std::chrono::high_resolution_clock::now();
			    auto elapsed_Time = std::chrono::duration_cast<std::chrono::nanoseconds>(CurTime - GlobalStartTime).count();
			    double elapsed_Time_Percent=elapsed_Time/TargetLatency;
			    elapsed_Task_Percent += taskPercentage;
			    int jump_freq = std::ceil(elapsed_Time_Percent/elapsed_Task_Percent);
			    CurGPUFreq += jump_freq;


			    //Balancing
			    double W = (U_GPU - U_CPU)/(U_GPU + U_CPU);
				if(std::abs(W) > Gamma){
					//Imbalanced
					std::cerr<<"Imbalanced\n";
					if(W > 0){
						//More load on GPU
						if (CurGPUFreq < 4){
							CurGPUFreq += 1;
						}
						else{
							CurBigFreq=std::max(0, CurBigFreq-1);
						}

					}
					else{
						//More load on CPU
						if (CurBigFreq < 7){
							CurBigFreq +=1;
						}
						else{
							CurGPUFreq = std::max(0, CurGPUFreq-1);
						}
					}

				}
				else{
					std::cerr<<"Balance CPU and GPU\n";
				}
				task.GPUFreq=CurGPUFreq;
				task.LittleFreq=CurLittleFreq;
				task.bigFreq=CurBigFreq;
				task.apply_freq(task.node->name());
			}
			//Reset the Freqs to initial values
			if(task.ending && last_layer){
				task.GPUFreq=CurGPUFreq=GFreqInit;
				task.LittleFreq=CurLittleFreq=LFreqInit;
				task.bigFreq=CurBigFreq=BFreqInit;
			}
			/**************************************************/

#endif


    		/*Profiling tasks
			task(nn);
    		if(task.ending && task.node->name()!=last_task_name){
    			if (-1 == GPIOWrite(POUT, 1)){
    				std::cerr<<"Could not write to GPIO\n";
    			}
    			task.apply_freq(task.node->name());
    			std::this_thread::sleep_for(std::chrono::milliseconds(8));
    		}*/
    		/*Profiling transfers
    		if(task.ending ){
    			task.apply_freq(task.node->name());
    			std::this_thread::sleep_for(std::chrono::milliseconds(12));
				if (-1 == GPIOWrite(POUT, 1)){
					std::cerr<<"Could not write to GPIO\n";
				}
			}*/
    		/*Profiling whole network **
    		task(nn);
			if(task.ending ){
				task.apply_freq(task.node->name());
			}
			/**************************/



#if streamline > 0
    		if(task.ending)
    			c=c+1;
    		ANNOTATE_CHANNEL_END(cc++);
#endif
    	}
        auto t0=std::chrono::high_resolution_clock::now();
        auto nanosec = t0.time_since_epoch();
#if My_print > 0
        std::cout<<"Executionhelpers, tasks() time: "<<nanosec.count()<<std::endl;
#endif
    }

    // Release memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->release();
        }
    }
}

bool call_all_output_node_accessors(ExecutionWorkload &workload)
{
    bool is_valid = true;
    std::for_each(std::begin(workload.outputs), std::end(workload.outputs), [&](Tensor * output_tensor)
    {
        bool valid_output = (output_tensor != nullptr) && output_tensor->my_call_accessor();
        is_valid          = is_valid && valid_output;
    });

    return is_valid;
}
} // namespace detail
} // namespace graph
} // namespace arm_compute
