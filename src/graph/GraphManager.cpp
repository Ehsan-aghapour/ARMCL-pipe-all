/*
 * Copyright (c) 2018-2019 Arm Limited.
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
//#include"annotate/Sr_ann.c"
//#include "utils/streamline_annotate.h"
#include "arm_compute/graph/printers/DotGraphPrinter.h"
#ifndef My_print
#include "arm_compute/gl_vs.h"
#endif

//#include "Power.h"


#include "arm_compute/graph/GraphManager.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/PassManager.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/detail/CrossLayerMemoryManagerHelpers.h"
#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include "arm_compute/graph/algorithms/TopologicalSort.h"

//#include "power.h"
#include "utils/Power.h"
#include "utils/DVFS.h"

namespace arm_compute
{
namespace graph
{

GraphManager::GraphManager()
    : _workloads()
{
	/*for(int i=0;i<20;i++){
		in[i]=out[i]=trans[i]=0;
	}*/
}

//#include "arm_compute/graph/backends/BackendRegistry.h"
int GraphManager::last_graph_id=0;
void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target, std::set<int> *blocking_set, int blocking)
{
    // Check if graph has been registered
	last_graph_id=graph.id();
	//std::cerr<<"finzing g id "<<graph.id()<<"-----------\n\n\n\n";
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }
    //std::cout<<"graph id:"<<graph.id()<<std::endl;
    // Apply IR mutating passes
    //std::cerr<<"befor pass 1 graph "<<graph.id()<<std::endl;
    //print_times(graph,1);

    pm.run_type(graph, IGraphMutator::MutationType::IR);
    // Force target to all graph construct
    // TODO (COMPMID-2014) : Support heterogeneous execution
    Target forced_target = target;
    if(!is_target_supported(target))
    {
	//Ehsan
	//std::cout<<"target is not supported."<<std::endl;

        forced_target = get_default_target();
        ARM_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
    }
#if My_print > 0
    //Ehsan
    std::cout<<"*********force target is: "<<target<<std::endl;
#endif
    force_target_to_graph(graph, forced_target);


    // Setup backend context
    // TODO (COMPMID-2014) : Setup all backends needed by the graph

    setup_requested_backend_context(ctx, forced_target);
    // Configure all tensors
    /*Ehsan:
     * set TensforHandle for all tensors which TensorInfo of TensorAllocator for each TensorHandle is set based on information of each tensor such as shape,datatype,
     * quantinfo and ...
     * strides in bytes for all dimensions also is set in tensorInfo
     */
    detail::configure_all_tensors(graph);
    // Apply backend mutating passes

    //std::cerr<<"befor pass 2 graph "<<graph.id()<<std::endl;
    //print_times(graph,1);
    pm.run_type(graph, IGraphMutator::MutationType::Backend);
    // Perform topological sort
    std::vector<NodeID> topological_sorted_nodes = dfs(graph);
    // Validate all nodes
    detail::validate_all_nodes(graph);

    // Configure all nodes
    auto workload = detail::configure_all_nodes(graph, ctx, topological_sorted_nodes);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");
#if My_print > 0
    //Ehsan
    std::cout<<"\nGraphManager, outputs size:"<<workload.outputs.size()<<std::endl;
#endif
    // Allocate const tensors and call accessors
    detail::allocate_const_tensors(graph);
    detail::call_all_const_node_accessors(graph);
    // Prepare graph
    detail::prepare_all_tasks(workload);

    //Ehsan
    int ii=0;
    //std::set<int> blocking_set1 {1, 2, 3, 4};
    //std::set<int> *blocking_set=&blocking_set1;
    std::vector<std::string> task_names;
    std::stringstream ss;
    std::stringstream ss2;
    ss<<"std::string task_names[] = { ";
    ss2<<"std::set<std::string> end_task_names = { ";
    int eccc=0;
    int ccc=0;

    for(auto &task : workload.tasks)
    {
    	if(!task.task)
    		continue;

    	//std::cerr<<eccc<<"( "<<ccc<<" ):"<<task.node->name()<<std::endl;

    	ss<<"\""<<task.node->name()<<"\", ";
    	ccc++;
    	//std::cerr<<"Task Name: "<<task.node->name()<<std::endl;
    	task_names.push_back(task.node->name());

    	bool b=false;
    	if(blocking_set->find(ii) != blocking_set->end()){
    	      b=true;
    	      task.ending=true;
    	      eccc++;
    	      ss2<<"\""<<task.node->name()<<"\", ";

    	}

    	if(blocking==1){
    		if(blocking_set!=NULL and b && target==arm_compute::graph::Target ::CL)
    		    task.block=1;
    	}
    	if(blocking==2){
    		if(blocking_set!=NULL && target==arm_compute::graph::Target ::CL){
    			task.block=1;
    		}
    	}

    	ii++;
    }

    ss.seekp(-2, std::ios_base::end);
    ss<<" };\n";
    std::cerr<<ss.str();
    std::cerr<<"number of tasks: "<<ccc<<std::endl;

    ss2.seekp(-2, std::ios_base::end);
	ss2<<" };\n";
	//std::cerr<<ss2.str();
	std::cerr<<"number of end tasks: "<<eccc<<std::endl;

#if My_print > 0
    //Ehsan
        DotGraphPrinter p;
        p.print(graph,std::cout);
#endif

    // Setup tensor memory (Allocate all tensors or setup transition manager)
    if(ctx.config().use_transition_memory_manager)
    {
#if My_print > 0
    	//Ehsan
    	std::cout<<"transition memory mangaer is used\n";
#endif

        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
        detail::allocate_all_tensors(graph);
    }
    // Finalize Graph context
    ctx.finalize();

    // Register graph
    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
    //std::cerr<<"after pass graph "<<graph.id()<<std::endl;
    //print_times(graph,1);
}
//Ehsan

void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target, std::set<std::string> *blocking_set, int blocking)
{
	last_graph_id=graph.id();
	//std::cerr<<"finzing g id "<<graph.id()<<"-----------\n\n\n\n";
    // Check if graph has been registered
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }
    //std::cout<<"graph id:"<<graph.id()<<std::endl;
    // Apply IR mutating passes
    //std::cerr<<"befor pass 1 graph "<<graph.id()<<std::endl;
    //print_times(graph,1);

    pm.run_type(graph, IGraphMutator::MutationType::IR);
    // Force target to all graph construct
    // TODO (COMPMID-2014) : Support heterogeneous execution
    Target forced_target = target;
    if(!is_target_supported(target))
    {
	//Ehsan
	//std::cout<<"target is not supported."<<std::endl;

        forced_target = get_default_target();
        ARM_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
    }
#if My_print > 0
    //Ehsan
    std::cout<<"*********force target is: "<<target<<std::endl;
#endif
    force_target_to_graph(graph, forced_target);


    // Setup backend context
    // TODO (COMPMID-2014) : Setup all backends needed by the graph

    setup_requested_backend_context(ctx, forced_target);
    // Configure all tensors
    /*Ehsan:
     * set TensforHandle for all tensors which TensorInfo of TensorAllocator for each TensorHandle is set based on information of each tensor such as shape,datatype,
     * quantinfo and ...
     * strides in bytes for all dimensions also is set in tensorInfo
     */

    detail::configure_all_tensors(graph);

    // Apply backend mutating passes

    //std::cerr<<"befor pass 2 graph "<<graph.id()<<std::endl;
    //print_times(graph,1);
    pm.run_type(graph, IGraphMutator::MutationType::Backend);
    // Perform topological sort
    std::vector<NodeID> topological_sorted_nodes = dfs(graph);
    // Validate all nodes
    detail::validate_all_nodes(graph);
    // Configure all nodes
    auto workload = detail::configure_all_nodes(graph, ctx, topological_sorted_nodes);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");
#if My_print > 0
    //Ehsan
    std::cout<<"\nGraphManager, outputs size:"<<workload.outputs.size()<<std::endl;
#endif
    // Allocate const tensors and call accessors
    detail::allocate_const_tensors(graph);
    detail::call_all_const_node_accessors(graph);
    // Prepare graph
    detail::prepare_all_tasks(workload);

    //Ehsan
    int ii=0;
    //std::set<int> blocking_set1 {1, 2, 3, 4};
    //std::set<int> *blocking_set=&blocking_set1;
    std::vector<std::string> task_names;
    std::stringstream ss;
    std::stringstream ss2;
    std::stringstream ss3;
    ss<<"std::string task_names[] = { ";
    ss2<<"std::set<std::string> end_task_names = { ";
    int eccc=0;
    int ccc=0;
    ss3<<"std::set<std::string> start_task_names = { ";
    bool starting=true;
    for(auto &task : workload.tasks)
    {
    	if(!task.task)
    		continue;

    	if(starting){
    		ss3<<"\""<<task.node->name()<<"\", ";
    		task.starting=true;
    		//task.set_freq(ctx.config().freq.at(task.node->name()), ctx.config().cluster);
    		//task.set_freq(ctx.config().freq.at(task.node->name()).second, ctx.config().freq.at(task.node->name()).first);
    		starting=false;
    	}
    	//std::cerr<<eccc<<"( "<<ccc<<" ):"<<task.node->name()<<std::endl;

    	ss<<"\""<<task.node->name()<<"\", ";
    	ccc++;
    	//std::cerr<<"Task Name: "<<task.node->name()<<std::endl;
    	task_names.push_back(task.node->name());

    	bool b=false;
    	if(blocking_set->find(task.node->name()) != blocking_set->end()){
    		  //task.set_freq(ctx.config().freq.at(task.node->name()));
    	      b=true;
    	      starting=true;
    	      task.ending=true;
    	      eccc++;
    	      ss2<<"\""<<task.node->name()<<"\", ";

    	}

    	if(blocking==1){
    		if(blocking_set!=NULL and b && target==arm_compute::graph::Target ::CL)
    		    task.block=1;
    	}
    	if(blocking==2){
    		if(blocking_set!=NULL && target==arm_compute::graph::Target ::CL){
    			task.block=1;
    		}
    	}

    	ii++;
    }

    ss.seekp(-2, std::ios_base::end);
    ss<<" };\n";
    std::cerr<<ss.str();
    std::cerr<<"number of tasks: "<<ccc<<std::endl;

    ss2.seekp(-2, std::ios_base::end);
	ss2<<" };\n";
	std::cerr<<ss2.str();
	std::cerr<<"number of end tasks: "<<eccc<<std::endl;

	ss3.seekp(-2, std::ios_base::end);
	ss3<<" };\n";
	std::cerr<<ss3.str();
	std::cerr<<"number of start tasks: "<<eccc<<std::endl;

#if My_print > 0
    //Ehsan
        DotGraphPrinter p;
        p.print(graph,std::cout);
#endif

    // Setup tensor memory (Allocate all tensors or setup transition manager)
    if(ctx.config().use_transition_memory_manager)
    {
#if My_print > 0
    	//Ehsan
    	std::cout<<"transition memory mangaer is used\n";
#endif

        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
        detail::allocate_all_tensors(graph);
    }
    //std::cerr<<"sf\n";
    // Finalize Graph context
    ctx.finalize();

    // Register graph
    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
    std::cerr<<"after pass graph "<<graph.id()<<std::endl;
    //print_times(graph,1);
}
//Ehsan

void GraphManager::print_times(Graph &graph, int n,std::vector<std::string> endings)
{
	auto it = _workloads.find(graph.id());
	ExecutionWorkload *workload = &it->second;
	std::string first_ending;
	for(auto &task:workload->tasks){
		if (task.ending){
			first_ending=task.node->name();
			break;
		}
	}
	auto iter = std::find(endings.begin(), endings.end(), first_ending);
	int l=std::distance(endings.begin(), iter);
	int s=l;
	//std::cerr<<"\n\n\n\n\n\nprint times\n";
	//std::cerr<<graph.id()<<"\t"<<workload->tasks[0].node->name()<<


	double sum=0;
	int c=0;

	double tt=0;
	for(auto &task:workload->tasks){
		if(!task.task){
			std::cerr<<"nadareeeeeeeee\n";
			continue;
		}
		std::cout<<c++<<"\tLayer Name: "<<task.node->name()
				<<" \t Layer time: "<<task.time(n)
				<<" \t number of inputs: "<<task.node->num_inputs()
				<<" \t input shape: "<<task.node->input(0)->desc().shape
				<<" \t output shape: "<<task.node->output(0)->desc().shape<<std::endl;

		tt+=task.time(n);
		if(task.ending){
			std::cout<<"Layer Number: "<<l<<" \t time: "<<tt<<std::endl;
			tt=0;
			l++;
			std::cout<<"----------------------------\n";
		}
		sum+=task.time(n);
	}
	std::cout<<"\n Sum of Layers time: "<<sum<<std::endl;
	std::cout<<"\n input time of layer "<<s<<" : "<<1000*input_time/double(n)<<std::endl;
	std::cout<<"\n transfer_time of layer "<<s<<" : "<<1000*transfer_time/double(n)<<std::endl;
	std::cout<<"\n output time of layer "<<l-1<<" : "<<1000*output_time/double(n)<<std::endl;
	std::cout<<"\n this part task time: "<<1000*task_time/double(n)<<std::endl;

	/*for(int i=0;i<20;i++){
		std::cout<<i<<"-- in:"<<1000*in[i]<<"   trans:"<<1000*trans[i]<<" out:"<<1000*out[i]<<std::endl;
	}*/
}

void GraphManager::print_lw_config(Graph &graph){
	auto it = _workloads.find(graph.id());
	ExecutionWorkload *workload = &it->second;
	for(auto &task:workload->tasks){
		std::cerr<<"task: "<<task.node->name()<<" set freq for next task to "<<task.LittleFreq<<", "<<task.bigFreq<<", "<<task.GPUFreq<<std::endl;
	}
}

void GraphManager::reset(Graph &graph)
{
	auto it = _workloads.find(graph.id());
	ExecutionWorkload *workload = &it->second;
	for(auto &task:workload->tasks){
		task.reset();
	}
	input_time=0;
	task_time=0;
	output_time=0;
	transfer_time=0;
}

void GraphManager::set_tasks_freqs(Graph &graph,std::map<std::string, std::array<int, 3>> freq_layer){
	auto &workload=_workloads[graph.id()];
	for(auto &task : workload.tasks)
	    {
	    	if(!task.task)
	    		continue;

	    	if(task.ending){
	    		task.set_freq(freq_layer.at(task.node->name()));
	    	}
	    }
}

/*
void GraphManager::execute_graph(Graph &graph)
{
    // Check if graph is finalized
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
//Ehsan measure input, task and output timings:
    while(true)
    {
        // Call input accessors
        //double tot=0;

        //auto tstart=std::chrono::high_resolution_clock::now();
        if(!detail::call_all_input_node_accessors(it->second))
        {
            return;
        }
        //auto tfinish=std::chrono::high_resolution_clock::now();
        //in += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        //std::cout<<"Input accessor duration: "<<Cost0<<std::endl;
        // Run graph
        detail::call_all_tasks(it->second);
        //tstart=std::chrono::high_resolution_clock::now();
        //task += std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();

        //std::cout<<"task duration: "<<Cost0<<std::endl;
        // Call output accessors
        if(!detail::call_all_output_node_accessors(it->second))
        {
            //tfinish=std::chrono::high_resolution_clock::now();
            //out = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
            //std::cout<<"__Output accessor duration: "<<Cost0<<std::endl;
            //std::cout<<"tot_(input+tasks+output):"<<tot<<std::endl;
            return;
        }
        //tfinish=std::chrono::high_resolution_clock::now();
        //out = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        //tot = in+task+out;
        //std::cout<<"Output accessor duration: "<<Cost0<<std::endl;
        //std::cout<<"tot_:"<<tot<<std::endl;

    }
}
*/
std::mutex GraphManager::mmtx; // create a mutex
bool GraphManager::First_time=new bool(true);
//std::chrono::time_point<std::chrono::high_resolution_clock>* GraphManager::Task_finish_time;
//std::chrono::time_point<std::chrono::high_resolution_clock>* GraphManager::Output_finish_time;
std::atomic<std::chrono::time_point<std::chrono::high_resolution_clock>> GraphManager::Task_finish_time;
std::atomic<std::chrono::time_point<std::chrono::high_resolution_clock>> GraphManager::Output_finish_time;

void update_task(std::chrono::time_point<std::chrono::high_resolution_clock> t){
	GraphManager::mmtx.lock();
	GraphManager::Task_finish_time.store(t);
	GraphManager::mmtx.unlock();
}
double calc_task(std::chrono::time_point<std::chrono::high_resolution_clock> t){
	double x=0;
	GraphManager::mmtx.lock();
	x=std::chrono::duration_cast<std::chrono::duration<double>>(t - GraphManager::Task_finish_time.load()).count();
	GraphManager::mmtx.unlock();
	return x;
}
void update_out(std::chrono::time_point<std::chrono::high_resolution_clock> t){
	GraphManager::mmtx.lock();
	GraphManager::Output_finish_time.store(t);
	GraphManager::mmtx.unlock();
}
double calc_out(std::chrono::time_point<std::chrono::high_resolution_clock> t){
	double x=0;
	GraphManager::mmtx.lock();
	x=std::chrono::duration_cast<std::chrono::duration<double>>(t - GraphManager::Output_finish_time.load()).count();
	GraphManager::mmtx.unlock();
	return x;
}
void GraphManager::execute_graph(Graph &graph, int nn)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    while(true)
    {

		//ANNOTATE_CHANNEL_COLOR(1,ANNOTATE_GREEN,"input");
		auto tstart=std::chrono::high_resolution_clock::now();
		//std::cerr<<"graph_id:"<<graph.id()<<"last id: "<<last_graph_id <<std::endl;
		//std::cerr<<graph.id()<<" heye1\n\n";
		if(!detail::call_all_input_node_accessors(it->second))
		{
			return;
		}
		//std::cerr<<graph.id()<<" heye2\n\n";
		auto tfinish=std::chrono::high_resolution_clock::now();
		//ANNOTATE_CHANNEL_END(1);
		//ANNOTATE_CHANNEL_COLOR(2,ANNOTATE_YELLOW,"task");
		//double x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - Output_finish_time.load()).count();
		double x1=calc_out(tfinish);

		if(graph.id()){
			//double xt=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - Task_finish_time.load()).count();
			double xt=calc_task(tfinish);
			//std::cerr<<graph.id()<<" tplus:"<<x1<<std::endl;
			transfer_time+=xt;
			//trans[frame]=xt;
			//Input of next graphs reach to the measurement point sooner then output of previous graph
			if (x1>xt){
				x1=0;
			}
		}
		//in[frame]=x1;
		input_time +=x1;
		//std::cerr<<graph.id()<<" before\n\n";
		detail::call_all_tasks(it->second,nn,graph.id()==last_graph_id, graph.name());
		tstart=std::chrono::high_resolution_clock::now();
		/*profile tasks mode
		if(graph.id()!=last_graph_id && nn){
			if (-1 == GPIOWrite(POUT, 1)){
				std::cerr<<"Could not write to GPIO\n";
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(4));
		}*/
		//Task_finish_time.store(tstart);
		update_task(tstart);
		//ANNOTATE_CHANNEL_END(2);
		//ANNOTATE_CHANNEL_COLOR(3,ANNOTATE_BLACK,"output");
		double x2=std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();
		task_time += x2;
		double x3=0;
		if(!detail::call_all_output_node_accessors(it->second))
		{
			tfinish=std::chrono::high_resolution_clock::now();
			//Output_finish_time.store(tfinish);
			update_out(tfinish);
			auto &task=it->second.tasks[it->second.tasks.size()-1];
			//Just the last layer apply freqs and GPIO after output
			if(graph.id()==last_graph_id ){
				task.apply_freq(task.node->name());
				if(nn){
					if (-1 == GPIOWrite(POUT, 1)){
						std::cerr<<"Could not write to GPIO\n";
					}
				}
			}
			//task.apply_freq(task.node->name());

			/*
			 * profiling task mode
			if(graph.id()==last_graph_id && nn){
				if (-1 == GPIOWrite(POUT, 1)){
					std::cerr<<"Could not write to GPIO\n";
				}

			}*/

			/*******************************************
			 * Profiling whole network
			 ******************************************
			if(graph.id()==last_graph_id && nn){
				if(graph.id()==last_graph_id && nn){
					if (-1 == GPIOWrite(POUT, 1)){
						std::cerr<<"Could not write to GPIO\n";
					}

				}
			}
			/*******************************************
			 ******************************************/

			x3=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
			//out[frame]=x3;
			output_time += x3;
			//frame++;
			//ANNOTATE_CHANNEL_END(3);
			return;
		}
		tfinish=std::chrono::high_resolution_clock::now();
		//Output_finish_time.store(tfinish);
		update_out(tfinish);
		x3=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		//out[frame]=x3;
		output_time +=x3;
		//frame++;
	}
}



//Ehsan
void GraphManager::execute_graph(Graph &graph, bool anotate, int nn)
{
	if(!anotate)
	{
		execute_graph(graph, nn);
		return;
	}
	/*
	//std::cerr<<"ajab\n";
    // Check if graph is finalized
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    //Ehsan measure input, task and output timings:
    ANNOTATE_SETUP;
    if(anotate)
        ANNOTATE_MARKER_STR("start_running");
    static int test=0;
    static int cc=0;
    while(true)
    {
        // Call input accessors
        //double tot=0;

        ANNOTATE_CHANNEL_COLOR(cc,ANNOTATE_GREEN,"input");
#if My_print > 0
        std::cout<<"test is: "<<test++<<std::endl;
#endif
        auto tstart=std::chrono::high_resolution_clock::now();
        if(!detail::call_all_input_node_accessors(it->second))
        {
            auto tfinish=std::chrono::high_resolution_clock::now();
            ANNOTATE_CHANNEL_END(cc++);
#if My_print > 0
            std::cout<<"in: "<<in<<'\t';
#endif
            input_time += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
#if My_print > 0
            std::cout<<"updated in: "<<in<<std::endl;
            std::cout<<"input exit\n";
#endif
            return;
        }
        auto tfinish=std::chrono::high_resolution_clock::now();
        ANNOTATE_CHANNEL_END(cc++);
#if My_print > 0
        std::cout<<"in: "<<in<<'\t';
#endif
        input_time += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
#if My_print > 0
        std::cout<<"updated in: "<<in<<std::endl;
        //std::cout<<"Input accessor duration: "<<Cost0<<std::endl;
#endif
        // Run graph
        ANNOTATE_CHANNEL_COLOR(cc,ANNOTATE_YELLOW,"task");
        tfinish=std::chrono::high_resolution_clock::now();
        detail::call_all_tasks(it->second);
        tstart=std::chrono::high_resolution_clock::now();

        //std::cout<<"task_previous:"<<task<<std::endl;

        ANNOTATE_CHANNEL_END(cc++);
        ANNOTATE_CHANNEL_COLOR(cc,ANNOTATE_BLACK,"output");
#if My_print > 0
        std::cout<<"task: "<<task<<'\t';
#endif
        task_time += std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();
#if My_print > 0
        std::cout<<"updated task: "<<task<<std::endl;
        //std::cout<<"task duration: "<<task<<std::endl;
#endif
        // Call output accessors
        tstart=std::chrono::high_resolution_clock::now();
        if(!detail::call_all_output_node_accessors(it->second))
        {
            tfinish=std::chrono::high_resolution_clock::now();
#if My_print > 0
            std::cout<<"out: "<<out<<'\t';
#endif
            output_time += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
#if My_print > 0
            std::cout<<"updated out: "<<out<<std::endl;
             std::cout<<"__Output accessor duration: "<<out<<std::endl;
            //std::cout<<"tot_(input+tasks+output):"<<tot<<std::endl;
#endif
            ANNOTATE_CHANNEL_END(cc++);
	    ANNOTATE_MARKER_STR("Finished...");
            return;
        }
        tfinish=std::chrono::high_resolution_clock::now();
        ANNOTATE_CHANNEL_END(cc++);
	//ANNOTATE_MARKER_STR("Finished");
#if My_print > 0
        std::cout<<"out: "<<out<<'\t';
#endif
        output_time += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
#if My_print > 0
        std::cout<<"updated out: "<<out<<std::endl;
        //tot = in+task+out;
        //std::cout<<"Output accessor duration: "<<out<<std::endl;
        //std::cout<<"tot_:"<<tot<<std::endl;
#endif
    }
    */
}

void GraphManager::invalidate_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    _workloads.erase(it);
}
} // namespace graph
} // namespace arm_compute
