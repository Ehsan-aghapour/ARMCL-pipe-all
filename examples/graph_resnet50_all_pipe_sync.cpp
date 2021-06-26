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


int core0=0;
int core1=0;
int core2=0;
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

/** Example demonstrating how to implement ResNetV1_50 network using the Compute Library's graph API */
class GraphResNetV1_50Example : public Example
{
public:
    GraphResNetV1_50Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "ResNetV1_50"), common_params2(), graph2(1,"ResNetV1_50_2"), common_params3(), graph3(1,"ResNetV1_50_3")
    {
    }



    void switch_graph(){
        // Finalize graph
        GraphConfig config;
        if(cluster==0){
        	config.num_threads = common_params.threads2;
        	//config.cluster=0;
        }
        else{
        	config.num_threads = common_params.threads;
        	//config.cluster=1;
        }
        //std::cout<<"n trds:"<<config.num_threads<<std::endl;
        config.cluster=cluster;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;

        //config.cluster=0;


        graph.finalize(common_params.target, config);

        //arm_compute::graph::Tensor *f_out;
        for(auto &node : graph.graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
            {
                f_out=node->input(0);
                 continue;
            }
        }

#if My_print > 0
        std::cout<<"Setup Second Graph,\n"<<f_out->desc().shape<<std::endl;
        //std::cout<<f_out->desc().shape.x()<<','<<f_out->desc().shape.y()<<','<<f_out->desc().shape.z()<<std::endl;
#endif

        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core1,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
        // Create input descriptor
        //const auto        operation_layout = common_params.data_layout;
        ////const TensorShape tensor_shape2     = permute_shape(TensorShape(13U, 13U, 384U, 1U), DataLayout::NCHW, operation_layout);
        const TensorShape tensor_shape2 = f_out->desc().shape;
        const auto        operation_layout = common_params.data_layout;
        input_descriptor2 = TensorDescriptor(tensor_shape2, common_params.data_type).set_layout(operation_layout);


    }

    void switch_graph2(){
        // Finalize graph
        GraphConfig config;

        if(cluster2==0){
        	//config.num_threads = common_params.threads2;
        	config.cluster = 0;
        }
        else{
        	//config.num_threads = common_params.threads;
        	config.cluster = 1;
        }
        config.cluster=cluster2;
        //config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;

        //config.cluster=0;


        graph2.finalize(common_params2.target, config);

        //arm_compute::graph::Tensor *f_out;
        for(auto &node : graph2.graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Output)
            {
                s_out=node->input(0);
                continue;
            }
        }

        for(auto &node : graph2.graph().nodes())
		{
			if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
			{
				s_in=node->output(0);
				continue;
			}
		}

#if My_print > 0
        std::cout<<"Setup Third Graph,\n"<<s_out->desc().shape<<std::endl;
        //std::cout<<f_out->desc().shape.x()<<','<<f_out->desc().shape.y()<<','<<f_out->desc().shape.z()<<std::endl;
#endif

        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core2,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
        // Create input descriptor
        //const auto        operation_layout = common_params.data_layout;
        ////const TensorShape tensor_shape2     = permute_shape(TensorShape(13U, 13U, 384U, 1U), DataLayout::NCHW, operation_layout);
        const TensorShape tensor_shape3 = s_out->desc().shape;
        const auto        operation_layout = common_params2.data_layout;
        input_descriptor3 = TensorDescriptor(tensor_shape3, common_params2.data_type).set_layout(operation_layout);


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
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<CaffePreproccessor>(mean_rgb,
                                                                                           false /* Do not convert to BGR */);

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, operation_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;


        //Ehsan
        //**********************************************************************************
        common_params2 = consume_common_graph_parameters(common_opts);

        common_params3 = consume_common_graph_parameters(common_opts);

        //common_params.target=arm_compute::graph::Target ::CL;
        //common_params2.target=arm_compute::graph::Target ::NEON;
        //std::cout<<int(common_params.target)<<std::endl<<std::endl;

        ////common_params2.target=static_cast<arm_compute::graph::Target>(3-int(common_params.target));
        std::string order=common_params.order;
		if(order[0]=='B'){
			common_params.target=static_cast<arm_compute::graph::Target>(1);
			cluster=1;
		}
		if(order[0]=='L'){
			common_params.target=static_cast<arm_compute::graph::Target>(1);
			cluster=0;
		}
		if(order[0]=='G'){
			common_params.target=static_cast<arm_compute::graph::Target>(2);
			cluster=2;
			gpu_index=0;
		}

		if(order[2]=='B'){
			common_params2.target=static_cast<arm_compute::graph::Target>(1);
			cluster2=1;
		}
		if(order[2]=='L'){
			common_params2.target=static_cast<arm_compute::graph::Target>(1);
			cluster2=0;
		}
		if(order[2]=='G'){
			common_params2.target=static_cast<arm_compute::graph::Target>(2);
			cluster2=2;
			gpu_index=1;
		}

		if(order[4]=='B'){
			common_params3.target=static_cast<arm_compute::graph::Target>(1);
			cluster3=1;
		}
		if(order[4]=='L'){
			common_params3.target=static_cast<arm_compute::graph::Target>(1);
			cluster3=0;
		}
		if(order[4]=='G'){
			common_params3.target=static_cast<arm_compute::graph::Target>(2);
			cluster3=2;
			gpu_index=2;
		}

		std::map<int, int> core = {{0, 1}, {1, 5}, {2, 4}};
		core0=core[cluster];
		core1=core[cluster2];
		core2=core[cluster3];

		//std::cout<<"\nFirst graph manages on core:"<<core0<<std::endl;
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(core0,&set);
		ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");


        common_params.labels="transfer_wait";
        if(order[0]=='G'){
        	common_params.labels="transfer";
        }
        common_params2.labels="transfer2_wait";
		if(order[2]=='G'){
			common_params2.labels="transfer2";
		}


        common_params2.image="transfer";
        if(order[0]=='G'){
        	common_params2.image="transfer_wait";
        }
        common_params3.image="transfer2";
        if(order[2]=='G'){
        	common_params3.image="transfer2_wait";
        }
        //common_params.threads=4;

        p=common_params.partition_point;
        p2=common_params.partition_point2;

        // Print parameter values
        //std::cout << common_params << std::endl;
        // Print parameter values
        // Print parameter values
        std::cout << common_params << std::endl;

        std::cout <<"\nGraph2:\n"<< common_params2 << std::endl;

        std::cout <<"\nGraph3:\n"<< common_params3 << std::endl;


        sub_graph=&graph;
        _common_params=&common_params;
        Layer=0;
        //bool second=false;
        annotate=common_params.annotate;
        save_model=common_params.save;

        //***************************************************************

        std::string test;

        (*sub_graph) << _common_params->target
              << _common_params->fast_math_hint;
        (*sub_graph)<< InputLayer(input_descriptor, get_input_accessor(*_common_params, std::move(preprocessor), false /* Do not convert to BGR */));
        (*sub_graph)<< ConvolutionLayer(
                  7U, 7U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_weights.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 3, 3))
              .set_name("conv1/convolution")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_beta.npy"),
                  0.0000100099996416f)
              .set_name("conv1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv1/Relu")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))).set_name("pool1/MaxPool");

        /*Layer++;
		if(Layer==p){
			common_params.labels="transfer";
			(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
			sub_graph=&graph2;
			_common_params=&common_params2;
			switch_graph();
			(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
			second=true;
		}*/


        add_residual_block(data_path, "block1", weights_layout, 64, 3, 2);


        /*Layer++;
		if(Layer==p){
			common_params.labels="transfer";
			(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
			sub_graph=&graph2;
			_common_params=&common_params2;
			switch_graph();
			(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
			second=true;
		}*/



        add_residual_block(data_path, "block2", weights_layout, 128, 4, 2);


        /*Layer++;
		if(Layer==p){
			common_params.labels="transfer";
			(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
			sub_graph=&graph2;
			_common_params=&common_params2;
			switch_graph();
			(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
			second=true;
		}*/

        add_residual_block(data_path, "block3", weights_layout, 256, 6, 2);


        /*Layer++;
		if(Layer==p){
			common_params.labels="transfer";
			(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
			sub_graph=&graph2;
			_common_params=&common_params2;
			switch_graph();
			(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
			second=true;
		}*/


        add_residual_block(data_path, "block4", weights_layout, 512, 3, 1);

        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, operation_layout)).set_name("pool5");



        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }
        if(Layer==p2){
        	//common_params2.labels="transfer2";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph3;
        	_common_params=&common_params3;
        	switch_graph2();
        	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
        	third=true;

        }

        if(!second){
        	common_params.labels=common_params3.labels;
        }
        else if(!third){
			common_params2.labels=common_params3.labels;
		}
        (*sub_graph)<< ConvolutionLayer(
                  1U, 1U, 1000U,
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_biases.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("logits/convolution")
              << FlattenLayer().set_name("predictions/Reshape")
              << SoftmaxLayer().set_name("predictions/Softmax")
              << OutputLayer(get_output_accessor(*_common_params, 5));


        // Finalize graph
        GraphConfig config;
        int total_graphs=second+third+1;
        if(total_graphs==3)
        	config.cluster = cluster3;
        if(total_graphs==2)
            config.cluster = cluster2;
        if(total_graphs==1)
            config.cluster = cluster;

        if(config.cluster==0){
        	config.num_threads = common_params.threads2;
        }
        else if(config.cluster==1){
        	config.num_threads = common_params.threads;
        }
        //config.num_threads = _common_params->threads;
        config.use_tuner   = _common_params->enable_tuner;
        config.tuner_mode  = _common_params->tuner_mode;
        config.tuner_file  = _common_params->tuner_file;
        config.mlgo_file   = _common_params->mlgo_file;
        //config.cluster=0;


        sub_graph->finalize(_common_params->target, config);


        //arm_compute::graph::Tensor *s_in;
        for(auto &node : sub_graph->graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
            {
            	if(third)
            		t_in = node->output(0);
            	else if(second)
            		s_in = node->output(0);
            }
        }

        std::cout<<"Partition layer:"<<p<<std::endl;
        std::cout<<"Second partition layer:"<<p2<<std::endl;
        std::cout<<"Total layers:"<<Layer+1<<std::endl<<std::endl;
        return true;
    }


private:
    void do_run() override
    {
        // Run graph
        //Ehsan
    	std::thread First(&GraphResNetV1_50Example::do_run_1,this,core0);
    	std::thread Second(&GraphResNetV1_50Example::do_run_2,this,core1);
    	std::thread Third(&GraphResNetV1_50Example::do_run_3,this,core2);
    	First.join();
    	Second.join();
    	Third.join();
    }


    void do_run_1(int core_id)
    {
        // Run graph
        //Ehsan
    	//int core_id=1;
/*
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core_id, &set);
        CPU_SET(5,&set);
        CPU_SET(4,&set);
        CPU_SET(3,&set);
        CPU_SET(2,&set);
        CPU_SET(1,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
*/


        cpu_set_t set;
        CPU_ZERO(&set);
        //CPU_SET(core_id, &set);
        //CPU_SET(1,&set);
        CPU_SET(core0,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");


        std::cout<<"start running first graph ...\n";
        ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
        double in=0;
        double task=0.0001;
        double out=0;
        int tt=(common_params.n);
        auto tstart=std::chrono::high_resolution_clock::now();
        //std::cout<<tstart.time_since_epoch().count()<<std::endl;
        //std::cout<<tt<<std::endl;
        for(int i=0;i<(tt+1);i++){
        	if(i==1){
        		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        		tstart=std::chrono::high_resolution_clock::now();
        		//std::cout<<tstart.time_since_epoch().count()<<std::endl;
        		in=task=out=0;
        		task=0;
        	}
			if(imgs){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				std::cout<<"\n\nFirst graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				im_acc->set_filename(images_list[image_index++]);
			}

            graph.run(in,task,out,annotate);
        }
        auto tfinish=std::chrono::high_resolution_clock::now();
        //std::cout<<tfinish.time_since_epoch().count()<<std::endl;
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=cost0/tt;
        in=in/tt;
        task=task/tt;
        out=out/tt;
        double tot=in+task+out;
        std::cout<<"\n\nCost:"<<Cost<<std::endl;
        std::cout<<"input_time:"<<in<<"\ntask_time:"<<task<<"\noutput_time:"<<out<<"\ntotal_time:"<<tot<<std::endl;
    }
    void do_run_2(int core_id)
    {
        // Run graph
        //Ehsan
    	//int core_id=1;


        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core1, &set);
        //CPU_SET(1,&set);
        //CPU_SET(0,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");

        std::cout<<"start running second graph ...\n";
        ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
        double in2=0;
        double task2=0;
        double out2=0;
        int tt=(common_params.n);
        auto tstart=std::chrono::high_resolution_clock::now();
        //std::cout<<tstart.time_since_epoch().count()<<std::endl;
        //std::cout<<tt<<std::endl;
        for(int i=0;i<(tt+1);i++){
        	if(i==1){
        		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        		tstart=std::chrono::high_resolution_clock::now();
        		//std::cout<<tstart.time_since_epoch().count()<<std::endl;
        		in2=task2=out2=0;
        	}
			if(imgs){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				std::cout<<"\n\nSecond graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				//im_acc->set_filename(images_list[image_index++]);
			}
			if(second)
			{
				graph2.run(in2,task2,out2,annotate);
			}
        }
        auto tfinish=std::chrono::high_resolution_clock::now();
        //std::cout<<tfinish.time_since_epoch().count()<<std::endl;
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=cost0/tt;
        std::cout<<"\n\nCost:"<<Cost<<std::endl;
        in2=in2/tt;
        task2=task2/tt;
        out2=out2/tt;
        double tot2=in2+task2+out2;
        //std::cout<<"Cost:"<<Cost<<std::endl;
        std::cout<<"\n\ninput2_time:"<<in2<<"\ntask2_time:"<<task2<<"\noutput2_time:"<<out2<<"\ntotal2_time:"<<tot2<<std::endl;
    }
    void do_run_3(int core_id)
        {
            // Run graph
            //Ehsan
        	//int core_id=1;


            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(core2, &set);
            //CPU_SET(1,&set);
            //CPU_SET(0,&set);
            ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");

            std::cout<<"start running third graph ...\n";
            ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
            double in3=0;
            double task3=0;
            double out3=0;
            int tt=(common_params.n);
            auto tstart=std::chrono::high_resolution_clock::now();
            //std::cout<<tstart.time_since_epoch().count()<<std::endl;
            //std::cout<<tt<<std::endl;
            for(int i=0;i<(tt+1);i++){
            	if(i==1){
            		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            		tstart=std::chrono::high_resolution_clock::now();
            		//std::cout<<tstart.time_since_epoch().count()<<std::endl;
            		in3=task3=out3=0;
            	}
    			if(imgs){
    				if(image_index>=images_list.size())
    						image_index=image_index%images_list.size();
    				std::cout<<"\n\nThird graph inferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
    				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
    				//im_acc->set_filename(images_list[image_index++]);
    			}
    			if(third)
    			{
    				graph3.run(in3,task3,out3,annotate);
    			}
            }
            auto tfinish=std::chrono::high_resolution_clock::now();
            //std::cout<<tfinish.time_since_epoch().count()<<std::endl;
            double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
            double Cost=cost0/tt;
            std::cout<<"\n\nCost:"<<Cost<<std::endl;
            in3=in3/tt;
            task3=task3/tt;
            out3=out3/tt;
            double tot3=in3+task3+out3;
            //std::cout<<"Cost:"<<Cost<<std::endl;
            std::cout<<"\n\ninput3_time:"<<in3<<"\ntask3_time:"<<task3<<"\noutput3_time:"<<out3<<"\ntotal3_time:"<<tot3<<std::endl;
        }


private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    CommonGraphParams  common_params2;
    Stream			   graph2;
    TensorDescriptor   input_descriptor2;
    bool     		   second=false;

    CommonGraphParams  common_params3;
    Stream			   graph3;
    TensorDescriptor   input_descriptor3;
    bool     		   third=false;

    Stream *sub_graph=&graph;
    CommonGraphParams *_common_params=&common_params;
    int Layer=0;
    int p=0;
    int p2=0;
    bool			   annotate{false};
    int gpu_index=0;
    int cluster,cluster2,cluster3 = 0;


    void add_residual_block(const std::string &data_path, const std::string &name, DataLayout weights_layout,
                            unsigned int base_depth, unsigned int num_units, unsigned int stride)
    {
        for(unsigned int i = 0; i < num_units; ++i)
        {


            Layer++;
            if(Layer==p){
            	//common_params.labels="transfer";
            	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
            	sub_graph=&graph2;
            	_common_params=&common_params2;
            	switch_graph();
            	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
            	second=true;
            }
            if(Layer==p2){
            	//common_params2.labels="transfer2";
            	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
            	sub_graph=&graph3;
            	_common_params=&common_params3;
            	switch_graph2();
            	(*sub_graph)<<InputLayer(input_descriptor3, get_input_accessor(*_common_params));
            	third=true;

            }



            std::stringstream unit_path_ss;
            unit_path_ss << "/cnn_data/resnet50_model/" << name << "_unit_" << (i + 1) << "_bottleneck_v1_";
            std::stringstream unit_name_ss;
            unit_name_ss << name << "/unit" << (i + 1) << "/bottleneck_v1/";

            std::string unit_path = unit_path_ss.str();
            std::string unit_name = unit_name_ss.str();

            unsigned int middle_stride = 1;

            if(i == (num_units - 1))
            {
                middle_stride = stride;
            }

            SubStream right(*sub_graph);
            right << ConvolutionLayer(
                      1U, 1U, base_depth,
                      get_weights_accessor(data_path, unit_path + "conv1_weights.npy", weights_layout),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(1, 1, 0, 0))
                  .set_name(unit_name + "conv1/convolution")
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_beta.npy"),
                      0.0000100099996416f)
                  .set_name(unit_name + "conv1/BatchNorm")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(
                      3U, 3U, base_depth,
                      get_weights_accessor(data_path, unit_path + "conv2_weights.npy", weights_layout),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(middle_stride, middle_stride, 1, 1))
                  .set_name(unit_name + "conv2/convolution")
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_beta.npy"),
                      0.0000100099996416f)
                  .set_name(unit_name + "conv2/BatchNorm")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(
                      1U, 1U, base_depth * 4,
                      get_weights_accessor(data_path, unit_path + "conv3_weights.npy", weights_layout),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(1, 1, 0, 0))
                  .set_name(unit_name + "conv3/convolution")
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_beta.npy"),
                      0.0000100099996416f)
                  .set_name(unit_name + "conv2/BatchNorm");

            if(i == 0)
            {
                SubStream left(*sub_graph);
                left << ConvolutionLayer(
                         1U, 1U, base_depth * 4,
                         get_weights_accessor(data_path, unit_path + "shortcut_weights.npy", weights_layout),
                         std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                         PadStrideInfo(1, 1, 0, 0))
                     .set_name(unit_name + "shortcut/convolution")
                     << BatchNormalizationLayer(
                         get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_moving_mean.npy"),
                         get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_moving_variance.npy"),
                         get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_gamma.npy"),
                         get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_beta.npy"),
                         0.0000100099996416f)
                     .set_name(unit_name + "shortcut/BatchNorm");

                *sub_graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(unit_name + "add");
            }
            else if(middle_stride > 1)
            {
                SubStream left(*sub_graph);
                left << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 1, common_params.data_layout, PadStrideInfo(middle_stride, middle_stride, 0, 0), true)).set_name(unit_name + "shortcut/MaxPool");

                *sub_graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(unit_name + "add");
            }
            else
            {
                SubStream left(*sub_graph);
                *sub_graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(unit_name + "add");
            }

            *sub_graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");
        }
    }
};

/** Main program for ResNetV1_50
 *
 * Model is based on:
 *      https://arxiv.org/abs/1512.03385
 *      "Deep Residual Learning for Image Recognition"
 *      Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
 *
 * Provenance: download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphResNetV1_50Example>(argc, argv);
}
