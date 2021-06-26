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

using namespace arm_compute;
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

/** Example demonstrating how to implement MobileNet's network using the Compute Library's graph API */
class GraphMobilenetExample : public Example
{
public:
    GraphMobilenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "MobileNetV1"), common_params2(), graph2(1,"MobileNetV1_2"), common_params3(), graph3(2,"MobileNetv1_3")
    {
        // Add model id option
        model_id_opt = cmd_parser.add_option<SimpleOption<int>>("model-id", 0);
        model_id_opt->set_help("Mobilenet model id (0: 1.0_224, else: 0.75_160");
    }
    GraphMobilenetExample(const GraphMobilenetExample &) = delete;
    GraphMobilenetExample &operator=(const GraphMobilenetExample &) = delete;
    ~GraphMobilenetExample() override                               = default;




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

        // Get model parameters
        int model_id = model_id_opt->value();

        // Create input descriptor
        unsigned int spatial_size = (model_id == 0 || common_params.data_type == DataType::QASYMM8) ? 224 : 160;

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(spatial_size, spatial_size, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);




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


        // Set graph hints
        graph << _common_params->target
              << _common_params->fast_math_hint;

        // Create core graph
        if(arm_compute::is_data_type_float(_common_params->data_type))
        {
            create_graph_float2(input_descriptor, model_id);
        }
        else
        {
            create_graph_qasymm(input_descriptor);
        }
        if(!second){
        	common_params.labels=common_params3.labels;
        }
        else if(!third){
			common_params2.labels=common_params3.labels;
		}
        // Create common tail
        (*sub_graph) << ReshapeLayer(TensorShape(1001U)).set_name("Reshape")
              << SoftmaxLayer().set_name("Softmax")
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
    void do_run() override
    {
        // Run graph
        //Ehsan
    	std::thread First(&GraphMobilenetExample::do_run_1,this,core0);
    	std::thread Second(&GraphMobilenetExample::do_run_2,this,core1);
    	std::thread Third(&GraphMobilenetExample::do_run_3,this,core2);
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
    SimpleOption<int> *model_id_opt{ nullptr };
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

    void create_graph_float(TensorDescriptor &input_descriptor, int model_id)
    {
        float       depth_scale = (model_id == 0) ? 1.f : 0.75;
        std::string model_path  = (model_id == 0) ? "/cnn_data/mobilenet_v1_1_224_model/" : "/cnn_data/mobilenet_v1_075_160_model/";

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<TFPreproccessor>();

        // Get trainable parameters data path
        std::string data_path = _common_params->data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }


        std::string w;
        std::cout<<"wwwww\n";
        std::cin>>w;
        //Ehsan
        Layer=0;

        (*sub_graph) << InputLayer(input_descriptor,
                            get_input_accessor(*_common_params, std::move(preprocessor), false))
              << ConvolutionLayer(
                  3U, 3U, 32U * depth_scale,
                  get_weights_accessor(data_path, "Conv2d_0_weights.npy", DataLayout::NCHW),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
              .set_name("Conv2d_0")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_beta.npy"),
                  0.001f)
              .set_name("Conv2d_0/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name("Conv2d_0/Relu6");


        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 2
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_1", 64 * depth_scale, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));

        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 3
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_2", 128 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));



        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 4
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_3", 128 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));




        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 5
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_4", 256 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));




        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 6
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_5", 256 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));




        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 7
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_6", 512 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));





        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 8
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_7", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));






        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 9
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_8", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));







        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 10
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_9", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));





        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 11
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_10", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));


        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
			(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
			sub_graph=&graph2;
			_common_params=&common_params2;
			switch_graph();
			(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
			second=true;
		}

		// Layer 12
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_11", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));



        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 13
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_12", 1024 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));




        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 14
        (*sub_graph) << get_dwsc_node_float(data_path, "Conv2d_13", 1024 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, _common_params->data_layout)).set_name("Logits/AvgPool_1a");




        Layer++;
        if(Layer==p){
        	common_params.labels="transfer";
        	(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
        	sub_graph=&graph2;
        	_common_params=&common_params2;
        	switch_graph();
        	(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
        	second=true;
        }

        // Layer 15
        (*sub_graph)<< ConvolutionLayer(
                  1U, 1U, 1001U,
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy", DataLayout::NCHW),
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_biases.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Logits/Conv2d_1c_1x1");
    }










    void create_graph_float2(TensorDescriptor &input_descriptor, int model_id)
    {
        float       depth_scale = (model_id == 0) ? 1.f : 0.75;
        std::string model_path  = (model_id == 0) ? "/cnn_data/mobilenet_v1_1_224_model/" : "/cnn_data/mobilenet_v1_075_160_model/";

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<TFPreproccessor>();

        // Get trainable parameters data path
        std::string data_path = _common_params->data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }



        //Ehsan
        Layer=0;

        (*sub_graph) << InputLayer(input_descriptor,
                            get_input_accessor(*_common_params, std::move(preprocessor), false))
              << ConvolutionLayer(
                  3U, 3U, 32U * depth_scale,
                  get_weights_accessor(data_path, "Conv2d_0_weights.npy", DataLayout::NCHW),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
              .set_name("Conv2d_0")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_beta.npy"),
                  0.001f)
              .set_name("Conv2d_0/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name("Conv2d_0/Relu6");

        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 2
        get_dwsc_node_float2(data_path, "Conv2d_1", 64 * depth_scale, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));

        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 4
        get_dwsc_node_float2(data_path, "Conv2d_2", 128 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));



        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 6
        get_dwsc_node_float2(data_path, "Conv2d_3", 128 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));




        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 8
        get_dwsc_node_float2(data_path, "Conv2d_4", 256 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));




        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 10
        get_dwsc_node_float2(data_path, "Conv2d_5", 256 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));




        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 12
        get_dwsc_node_float2(data_path, "Conv2d_6", 512 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));



        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 14
        get_dwsc_node_float2(data_path, "Conv2d_7", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));


        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 16
        get_dwsc_node_float2(data_path, "Conv2d_8", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));



        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 18
        get_dwsc_node_float2(data_path, "Conv2d_9", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));



        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 20
        get_dwsc_node_float2(data_path, "Conv2d_10", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));


        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

		// Layer 22
        get_dwsc_node_float2(data_path, "Conv2d_11", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));


        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 24
        get_dwsc_node_float2(data_path, "Conv2d_12", 1024 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));

        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 26
        get_dwsc_node_float2(data_path, "Conv2d_13", 1024 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, _common_params->data_layout)).set_name("Logits/AvgPool_1a");

        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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

        // Layer 28
        (*sub_graph)<< ConvolutionLayer(
                  1U, 1U, 1001U,
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy", DataLayout::NCHW),
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_biases.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Logits/Conv2d_1c_1x1");
    }















    void create_graph_qasymm(TensorDescriptor &input_descriptor)
    {
        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += "/cnn_data/mobilenet_qasymm8_model/";
        }

        // Quantization info taken from the AndroidNN QASYMM8 MobileNet example
        const QuantizationInfo in_quant_info = QuantizationInfo(0.0078125f, 128);

        const std::vector<QuantizationInfo> conv_weights_quant_info =
        {
            QuantizationInfo(0.02182667888700962f, 151), // conv0
            QuantizationInfo(0.004986600950360298f, 74)  // conv14
        };
        const std::vector<QuantizationInfo> conv_out_quant_info =
        {
            QuantizationInfo(0.023528477177023888f, 0), // conv0
            QuantizationInfo(0.16609922051429749f, 66)  // conv14
        };

        const std::vector<QuantizationInfo> depth_weights_quant_info =
        {
            QuantizationInfo(0.29219913482666016f, 110),  // dwsc1
            QuantizationInfo(0.40277284383773804f, 130),  // dwsc2
            QuantizationInfo(0.06053730100393295f, 160),  // dwsc3
            QuantizationInfo(0.01675807684659958f, 123),  // dwsc4
            QuantizationInfo(0.04105526953935623f, 129),  // dwsc5
            QuantizationInfo(0.013460792601108551f, 122), // dwsc6
            QuantizationInfo(0.036934755742549896f, 132), // dwsc7
            QuantizationInfo(0.042609862983226776f, 94),  // dwsc8
            QuantizationInfo(0.028358859941363335f, 127), // dwsc9
            QuantizationInfo(0.024329448118805885f, 134), // dwsc10
            QuantizationInfo(0.019366811960935593f, 106), // dwsc11
            QuantizationInfo(0.007835594937205315f, 126), // dwsc12
            QuantizationInfo(0.12616927921772003f, 211)   // dwsc13
        };

        const std::vector<QuantizationInfo> point_weights_quant_info =
        {
            QuantizationInfo(0.030420949682593346f, 121), // dwsc1
            QuantizationInfo(0.015148180536925793f, 104), // dwsc2
            QuantizationInfo(0.013755458407104015f, 94),  // dwsc3
            QuantizationInfo(0.007601846940815449f, 151), // dwsc4
            QuantizationInfo(0.006431614048779011f, 122), // dwsc5
            QuantizationInfo(0.00917122047394514f, 109),  // dwsc6
            QuantizationInfo(0.005300046876072884f, 140), // dwsc7
            QuantizationInfo(0.0049632852897048f, 127),   // dwsc8
            QuantizationInfo(0.007770895957946777f, 89),  // dwsc9
            QuantizationInfo(0.009658650495111942f, 99),  // dwsc10
            QuantizationInfo(0.005446993745863438f, 153), // dwsc11
            QuantizationInfo(0.00817922968417406f, 130),  // dwsc12
            QuantizationInfo(0.018048152327537537f, 95)   // dwsc13
        };

        graph << InputLayer(input_descriptor.set_quantization_info(in_quant_info),
                            get_input_accessor(common_params, nullptr, false))
              << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "Conv2d_0_weights.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_bias.npy"),
                  PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
                  1, conv_weights_quant_info.at(0), conv_out_quant_info.at(0))
              .set_name("Conv2d_0")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name("Conv2d_0/Relu6");
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_1", 64U, PadStrideInfo(1U, 1U, 1U, 1U), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(0), point_weights_quant_info.at(0));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_2", 128U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(1),
                                      point_weights_quant_info.at(1));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_3", 128U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(2),
                                      point_weights_quant_info.at(2));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_4", 256U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(3),
                                      point_weights_quant_info.at(3));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_5", 256U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(4),
                                      point_weights_quant_info.at(4));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_6", 512U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(5),
                                      point_weights_quant_info.at(5));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_7", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(6),
                                      point_weights_quant_info.at(6));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_8", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(7),
                                      point_weights_quant_info.at(7));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_9", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(8),
                                      point_weights_quant_info.at(8));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_10", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(9),
                                      point_weights_quant_info.at(9));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_11", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(10),
                                      point_weights_quant_info.at(10));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_12", 1024U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(11),
                                      point_weights_quant_info.at(11));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_13", 1024U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(12),
                                      point_weights_quant_info.at(12))
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, common_params.data_layout)).set_name("Logits/AvgPool_1a")
              << ConvolutionLayer(
                  1U, 1U, 1001U,
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy"),
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_bias.npy"),
                  PadStrideInfo(1U, 1U, 0U, 0U), 1, conv_weights_quant_info.at(1), conv_out_quant_info.at(1))
              .set_name("Logits/Conv2d_1c_1x1");
    }

    ConcatLayer get_dwsc_node_float(const std::string &data_path, std::string &&param_path,
                                    unsigned int  conv_filt,
                                    PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
    {
        std::string total_path = param_path + "_";
        SubStream   sg(*sub_graph);
        sg << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "depthwise_depthwise_weights.npy", DataLayout::NCHW),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               dwc_pad_stride_info)
           .set_name(total_path + "depthwise/depthwise")
           << BatchNormalizationLayer(
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_mean.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_variance.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_gamma.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_beta.npy"),
               0.001f)
           .set_name(total_path + "depthwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "depthwise/Relu6");

           sg << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "pointwise_weights.npy", DataLayout::NCHW),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               conv_pad_stride_info)
           .set_name(total_path + "pointwise/Conv2D")
           << BatchNormalizationLayer(
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_mean.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_variance.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_gamma.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_beta.npy"),
               0.001f)
           .set_name(total_path + "pointwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "pointwise/Relu6");

        return ConcatLayer(std::move(sg));
    }



    void get_dwsc_node_float2(const std::string &data_path, std::string &&param_path,
                                    unsigned int  conv_filt,
                                    PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
    {
        std::string total_path = param_path + "_";
        //SubStream   sg(*sub_graph);
        (*sub_graph) << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "depthwise_depthwise_weights.npy", DataLayout::NCHW),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               dwc_pad_stride_info)
           .set_name(total_path + "depthwise/depthwise")
           << BatchNormalizationLayer(
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_mean.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_variance.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_gamma.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_beta.npy"),
               0.001f)
           .set_name(total_path + "depthwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "depthwise/Relu6");

        Layer++;
        if(Layer==p){
        	//common_params.labels="transfer_wait";
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



        (*sub_graph) << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "pointwise_weights.npy", DataLayout::NCHW),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               conv_pad_stride_info)
           .set_name(total_path + "pointwise/Conv2D")
           << BatchNormalizationLayer(
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_mean.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_variance.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_gamma.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_beta.npy"),
               0.001f)
           .set_name(total_path + "pointwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "pointwise/Relu6");

        //return ConcatLayer(std::move(sg));
    }




    ConcatLayer get_dwsc_node_qasymm(const std::string &data_path, std::string &&param_path,
                                     const unsigned int conv_filt,
                                     PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info,
                                     QuantizationInfo depth_weights_quant_info, QuantizationInfo point_weights_quant_info)
    {
        std::string total_path = param_path + "_";
        SubStream   sg(graph);

        sg << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "depthwise_weights.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_bias.npy"),
               dwc_pad_stride_info, 1, std::move(depth_weights_quant_info))
           .set_name(total_path + "depthwise/depthwise")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name(total_path + "depthwise/Relu6")
           << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "pointwise_weights.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_bias.npy"),
               conv_pad_stride_info, 1, std::move(point_weights_quant_info))
           .set_name(total_path + "pointwise/Conv2D")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name(total_path + "pointwise/Relu6");

        return ConcatLayer(std::move(sg));
    }
};

/** Main program for MobileNetV1
 *
 * Model is based on:
 *      https://arxiv.org/abs/1704.04861
 *      "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
 *      Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
 *
 * Provenance: download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
 *             download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160.tgz
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphMobilenetExample>(argc, argv);
}
