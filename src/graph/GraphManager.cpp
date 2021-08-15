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
#include"annotate/Sr_ann.c"
#include "arm_compute/graph/printers/DotGraphPrinter.h"
#ifndef My_print
#include "arm_compute/gl_vs.h"
#endif


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

namespace arm_compute
{
namespace graph
{
GraphManager::GraphManager()
    : _workloads()
{
}

//#include "arm_compute/graph/backends/BackendRegistry.h"
void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target, std::set<int> *blocking_set, int blocking)
{
    // Check if graph has been registered
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }
    //std::cout<<"graph id:"<<graph.id()<<std::endl;
    // Apply IR mutating passes
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
    for(auto &task : workload.tasks)
    {
    	bool b=false;
    	if(blocking_set->find(ii) != blocking_set->end()){
    	      b=true;
    	      task.ending=true;
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
}
//Ehsan

void GraphManager::print_times(Graph &graph, int n)
{
	auto it = _workloads.find(graph.id());
	ExecutionWorkload *workload = &it->second;
	double sum=0;
	int c=0;
	int l=0;
	double tt=0;
	for(auto &task:workload->tasks){
		std::cout<<c++<<"\tLayer Name: "<<task.node->name()<<" \t Layer time: "<<task.time(n)<<std::endl;
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
}

void GraphManager::reset(Graph &graph)
{
	auto it = _workloads.find(graph.id());
	ExecutionWorkload *workload = &it->second;
	for(auto &task:workload->tasks){
		task.reset();
	}
}

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


void GraphManager::execute_graph(Graph &graph,double &in, double &task, double &out, int nn)
{
    // Check if graph is finalized
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    //Ehsan measure input, task and output timings:
    while(true)
    {
        // Call input accessors
	//double tot=0;
	//ANNOTATE_CHANNEL_COLOR(1,ANNOTATE_GREEN,"input");
	auto tstart=std::chrono::high_resolution_clock::now();
        if(!detail::call_all_input_node_accessors(it->second))
        {
            return;
        }
        //std::cout<<"call all input called\n";
	auto tfinish=std::chrono::high_resolution_clock::now();
	//ANNOTATE_CHANNEL_END(1);
	//ANNOTATE_CHANNEL_COLOR(2,ANNOTATE_YELLOW,"task");
	in += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
	//std::cout<<"Input accessor duration: "<<Cost0<<std::endl;
        // Run graph
	//std::cout<<"\ntask:"<<task<<std::endl;
	//if(!task)
    detail::call_all_tasks(it->second,nn);
    //std::cout<<"call all tasks called\n";
	tstart=std::chrono::high_resolution_clock::now();
        //std::cout<<"task_previous:"<<task<<std::endl;

	//ANNOTATE_CHANNEL_END(2);
	//ANNOTATE_CHANNEL_COLOR(3,ANNOTATE_BLACK,"output");
	//if(!task)
	task += std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();

	//std::cout<<"\n2task:"<<task<<std::endl;
	//std::cout<<"task duration: "<<task<<std::endl;
        // Call output accessors
        if(!detail::call_all_output_node_accessors(it->second))
        {
	    tfinish=std::chrono::high_resolution_clock::now();
	    out += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
            // std::cout<<"__Output accessor duration: "<<out<<std::endl;
	    //std::cout<<"tot_(input+tasks+output):"<<tot<<std::endl;
	    //ANNOTATE_CHANNEL_END(3);
            return;
        }
	tfinish=std::chrono::high_resolution_clock::now();
	out += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

	//tot = in+task+out;
	//std::cout<<"Output accessor duration: "<<out<<std::endl;
	//std::cout<<"tot_:"<<tot<<std::endl;
	
    }
}



//Ehsan
void GraphManager::execute_graph(Graph &graph,double &in, double &task, double &out, bool anotate, int nn)
{
	if(!anotate)
	{
		execute_graph(graph, in, task, out, nn);
		return;
	}
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
            in += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
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
        in += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
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
        task += std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();
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
            out += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
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
        out += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
#if My_print > 0
        std::cout<<"updated out: "<<out<<std::endl;
        //tot = in+task+out;
        //std::cout<<"Output accessor duration: "<<out<<std::endl;
        //std::cout<<"tot_:"<<tot<<std::endl;
#endif
    }
}


void GraphManager::invalidate_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    _workloads.erase(it);
}
} // namespace graph
} // namespace arm_compute
