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


void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target)
{
    // Check if graph has been registered
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }

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
    //Ehsan
    std::cout<<"*********force target is: "<<target<<std::endl;
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

    //Ehsan
    std::cout<<"\nGraphManager, outputs size:"<<workload.outputs.size()<<std::endl;

    // Allocate const tensors and call accessors
    detail::allocate_const_tensors(graph);
    detail::call_all_const_node_accessors(graph);

    // Prepare graph
    detail::prepare_all_tasks(workload);

    //Ehsan
        DotGraphPrinter p;
        p.print(graph,std::cout);

    // Setup tensor memory (Allocate all tensors or setup transition manager)
    if(ctx.config().use_transition_memory_manager)
    {
    	//Ehsan
    	std::cout<<"transition memory mangaer is used\n";

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


void GraphManager::execute_graph(Graph &graph,double &in, double &task, double &out)
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
	auto tfinish=std::chrono::high_resolution_clock::now();
	//ANNOTATE_CHANNEL_END(1);
	//ANNOTATE_CHANNEL_COLOR(2,ANNOTATE_YELLOW,"task");
	in += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
	//std::cout<<"Input accessor duration: "<<Cost0<<std::endl;
        // Run graph
        detail::call_all_tasks(it->second);
	tstart=std::chrono::high_resolution_clock::now();

        //std::cout<<"task_previous:"<<task<<std::endl;

	//ANNOTATE_CHANNEL_END(2);
	//ANNOTATE_CHANNEL_COLOR(3,ANNOTATE_BLACK,"output");
	task += std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();

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
void GraphManager::execute_graph(Graph &graph,double &in, double &task, double &out, bool anotate)
{
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
        std::cout<<"test is: "<<test++<<std::endl;
        auto tstart=std::chrono::high_resolution_clock::now();
        if(!detail::call_all_input_node_accessors(it->second))
        {
            auto tfinish=std::chrono::high_resolution_clock::now();
            ANNOTATE_CHANNEL_END(cc++);
            std::cout<<"in: "<<in<<'\t';
            in += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
            std::cout<<"updated in: "<<in<<std::endl;
            std::cout<<"input exit\n";
            return;
        }
        auto tfinish=std::chrono::high_resolution_clock::now();
        ANNOTATE_CHANNEL_END(cc++);

        std::cout<<"in: "<<in<<'\t';
        in += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        std::cout<<"updated in: "<<in<<std::endl;
        //std::cout<<"Input accessor duration: "<<Cost0<<std::endl;
        // Run graph
        ANNOTATE_CHANNEL_COLOR(cc,ANNOTATE_YELLOW,"task");
        tfinish=std::chrono::high_resolution_clock::now();
        detail::call_all_tasks(it->second);
        tstart=std::chrono::high_resolution_clock::now();

        //std::cout<<"task_previous:"<<task<<std::endl;

        ANNOTATE_CHANNEL_END(cc++);
        ANNOTATE_CHANNEL_COLOR(cc,ANNOTATE_BLACK,"output");
        std::cout<<"task: "<<task<<'\t';
        task += std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();
        std::cout<<"updated task: "<<task<<std::endl;
        //std::cout<<"task duration: "<<task<<std::endl;
        // Call output accessors
        tstart=std::chrono::high_resolution_clock::now();
        if(!detail::call_all_output_node_accessors(it->second))
        {
            tfinish=std::chrono::high_resolution_clock::now();
            std::cout<<"out: "<<out<<'\t';
            out += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
            std::cout<<"updated out: "<<out<<std::endl;
             std::cout<<"__Output accessor duration: "<<out<<std::endl;
            //std::cout<<"tot_(input+tasks+output):"<<tot<<std::endl;
            ANNOTATE_CHANNEL_END(cc++);
	    ANNOTATE_MARKER_STR("Finished...");
            return;
        }
        tfinish=std::chrono::high_resolution_clock::now();
        ANNOTATE_CHANNEL_END(cc++);
	//ANNOTATE_MARKER_STR("Finished");
        std::cout<<"out: "<<out<<'\t';
        out += std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        std::cout<<"updated out: "<<out<<std::endl;
        //tot = in+task+out;
        //std::cout<<"Output accessor duration: "<<out<<std::endl;
        //std::cout<<"tot_:"<<tot<<std::endl;

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
