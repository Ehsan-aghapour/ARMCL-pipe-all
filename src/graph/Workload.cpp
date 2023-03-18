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
#include "arm_compute/graph/Workload.h"

#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorHandle.h"
#include "arm_compute/graph/nodes/PrintLayerNode.h"

//Ehsan
#include "arm_compute/runtime/CL/CLScheduler.h"
//std::map<std::string,double> task_times;


namespace arm_compute
{
namespace graph
{
void ExecutionTask::operator()()
{
    TaskExecutor::get().execute_function(*this);
}

void ExecutionTask::operator()(int nn)
{
    t+=TaskExecutor::get().execute_function2(*this,nn);
}

double ExecutionTask::time(int n){
	if(n!=0)
		return t/n;
	else
		return t;
}

void ExecutionTask::reset(){
	t=0;
	n=0;
}


void execute_task(ExecutionTask &task)
{
    if(task.task)
    {
    	//std::cerr<<"Node name:"<<task.node->name()<<"\t id:"<<task.node->id();
    	//auto start=std::chrono::high_resolution_clock::now();
        task.task->run();
        //arm_compute::CLScheduler::get().queue().finish();
        //auto finish=std::chrono::high_resolution_clock::now();
        //tt[task.node->name()]+=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count());
        //std::cout<<tt<<std::endl;
        //std::cerr<<"\t time: "<<1000*(std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count())<<std::endl;
    }
#ifdef ARM_COMPUTE_ASSERTS_ENABLED
    // COMPMID-3012 - Hide the printing logic from the execute_task method in the graph API
    else if(task.node->type() == NodeType::PrintLayer)
    {
        auto print_node   = dynamic_cast<PrintLayerNode *>(task.node);
        auto input_handle = print_node->input(0)->handle();
        auto transform    = print_node->transform();

        input_handle->map(true);
        ITensor *input = transform ? transform(&input_handle->tensor()) : &input_handle->tensor();
        input->print(print_node->stream(), print_node->format_info());
        input_handle->unmap();
    }
#endif // ARM_COMPUTE_ASSERTS_ENABLED
}

//std::map<std::string,double> task_times;
//void execute_task(ExecutionTask &task, const std::map<std::string,double>& tt)

double execute_task2(ExecutionTask &task, int nn)
{
	double t=0;
    if(task.task)
    {
    	//std::cerr<<"Node name:"<<task.node->name()<<"\t id:"<<task.node->id();
    	//static int n=-1;
    	//std::cerr<<"start running task with blockin and ending"<<task.block<<task.ending<<std::endl;
    	auto start=std::chrono::high_resolution_clock::now();
        task.task->run();
        if(task.block){
        	arm_compute::CLScheduler::get().queue().finish();
        	//std::cerr<<"block\n";
        }
        auto finish=std::chrono::high_resolution_clock::now();
        //std::cerr<<"task finished\n";
        t=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count());
        //task_times[task.node->name()]+=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count());
        //std::cerr<<"salam\n";
        //std::cout<<tt<<std::endl;
        //std::cerr<<"\t time: "<<1000*(std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count())<<std::endl;
    }
#ifdef ARM_COMPUTE_ASSERTS_ENABLED
    // COMPMID-3012 - Hide the printing logic from the execute_task method in the graph API
    else if(task.node->type() == NodeType::PrintLayer)
    {
        auto print_node   = dynamic_cast<PrintLayerNode *>(task.node);
        auto input_handle = print_node->input(0)->handle();
        auto transform    = print_node->transform();

        input_handle->map(true);
        ITensor *input = transform ? transform(&input_handle->tensor()) : &input_handle->tensor();
        input->print(print_node->stream(), print_node->format_info());
        input_handle->unmap();
    }
#endif // ARM_COMPUTE_ASSERTS_ENABLED
    return t;
}

void ExecutionTask::prepare()
{
    if(task)
    {
        task->prepare();
    }
}

TaskExecutor::TaskExecutor()
    : execute_function(execute_task),execute_function2(execute_task2)
{
}

TaskExecutor &TaskExecutor::get()
{
    static TaskExecutor executor;
    return executor;
}
} // namespace graph
} // namespace arm_compute
