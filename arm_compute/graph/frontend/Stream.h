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
#ifndef ARM_COMPUTE_GRAPH_STREAM_H
#define ARM_COMPUTE_GRAPH_STREAM_H

#include "arm_compute/graph/frontend/IStream.h"
#include "arm_compute/graph/frontend/IStreamOperators.h"
#include "arm_compute/graph/frontend/Types.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManager.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{
// Forward Declarations
class ILayer;

/** Stream frontend class to construct simple graphs in a stream fashion */
class Stream final : public IStream
{
public:
    /** Constructor
     *
     * @param[in] id   Stream id
     * @param[in] name Stream name
     */
    Stream(size_t id, std::string name);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Stream(const Stream &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Stream &operator=(const Stream &) = delete;
    /** Finalizes the stream for an execution target
     *
     * @param[in] target Execution target
     * @param[in] config (Optional) Graph configuration to use
     */
    void finalize(Target target, const GraphConfig &config, std::set<std::string> *b=NULL, int blocking=0);
    void finalize(Target target, const GraphConfig &config, std::set<int> *b, int blocking=0);

    /** Executes the stream **/
    //Ehsan
    void run(int nn=0);
    void run(bool annotate, int nn=0);

    void measure(int n,std::vector<std::string> endings={});
    void print_config();
    void set_tasks_freq(std::map<std::string, std::array<int, 3>> freq_layer){
    	_manager.set_tasks_freqs(_g,freq_layer);
    }
    void reset();
    // Inherited overridden methods
    void add_layer(ILayer &layer) override;
    Graph       &graph() override;
    const Graph &graph() const override;
    /*std::chrono::time_point<std::chrono::high_resolution_clock> get_start_time(){
    	return start;
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> get_finish_time(){
    	return finish;
    }
    void set_start_time(std::chrono::time_point<std::chrono::high_resolution_clock> t){
    	start=t;
    }
    void set_finish_time(std::chrono::time_point<std::chrono::high_resolution_clock> t){
    	finish=t;
    }
    double get_time(){
    	return std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();
    }*/

    void set_input_time(double t){
    	_manager.set_input_time(t);
    }
    void set_task_time(double t){
        _manager.set_task_time(t);
    }
    void set_output_time(double t){
        _manager.set_output_time(t);
    }
    void set_cost_time(double t){
    	cost=t;
    }
    void set_transfer_time(double t){
    	_manager.set_transfer_time(t);
    }

    double get_input_time(){
    	return _manager.get_input_time();
    }
    double get_task_time(){
    	return _manager.get_task_time();
    }
    double get_output_time(){
    	return _manager.get_output_time();
    }
    double get_cost_time(){
        return cost;
    }
    double get_transfer_time(){
    	return _manager.get_transfer_time();
    }


private:
    //Important: GraphContext must be declared *before* the GraphManager because the GraphManager
    //allocates resources from the context and therefore needs to be destroyed before the context during clean up.
    GraphContext _ctx;     /**< Graph context to use */
    GraphManager _manager; /**< Graph manager */
    Graph        _g;       /**< Internal graph representation of the stream */

    //Ehsan
    //std::chrono::time_point<std::chrono::high_resolution_clock> start;
    //std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    double input_time=0;
    double task_time=0;
    double output_time=0;
    double cost=0;
    double transfer_time=0;

};
} // namespace frontend
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_STREAM_H */
