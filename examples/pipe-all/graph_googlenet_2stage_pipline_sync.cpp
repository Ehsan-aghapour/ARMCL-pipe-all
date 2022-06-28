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
#include "annotate/streamline_annotate.h"

#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;


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

/** Example demonstrating how to implement Googlenet's network using the Compute Library's graph API */
class GraphGooglenetExample : public Example
{
public:
    GraphGooglenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GoogleNet"), common_params2(), graph2(1,"GoogleNet2")
    {
    }

    void switch_graph(){
        // Finalize graph
        GraphConfig config;

        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;


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


        // Create input descriptor
        //const auto        operation_layout = common_params.data_layout;
        ////const TensorShape tensor_shape2     = permute_shape(TensorShape(13U, 13U, 384U, 1U), DataLayout::NCHW, operation_layout);
        const TensorShape tensor_shape2 = f_out->desc().shape;
        const auto        operation_layout = common_params.data_layout;
        input_descriptor2 = TensorDescriptor(tensor_shape2, common_params.data_type).set_layout(operation_layout);


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

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

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
        common_params2 = consume_common_graph_parameters(common_opts);

        //common_params.target=arm_compute::graph::Target ::CL;
        //common_params2.target=arm_compute::graph::Target ::NEON;
        //std::cout<<int(common_params.target)<<std::endl<<std::endl;
        common_params2.target=static_cast<arm_compute::graph::Target>(3-int(common_params.target));

        common_params2.labels=common_params.labels;


        common_params2.image="transfer";
        //common_params.threads=4;

        p=common_params.partition_point;

        // Print parameter values
        //std::cout << common_params << std::endl;
        // Print parameter values
        std::cout <<"\nGraph2:\n"<< common_params2 << std::endl;


        sub_graph=&graph;
        _common_params=&common_params;
        Layer=0;
        //bool second=false;

        annotate=common_params.annotate;
        save_model=common_params.save;

        //***************************************************************


        (*sub_graph) << _common_params->target
              << _common_params->fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(*_common_params, std::move(preprocessor)))
              << ConvolutionLayer(
                  7U, 7U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv1/conv1_7x7_s2_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv1/conv1_7x7_s2_b.npy"),
                  PadStrideInfo(2, 2, 3, 3))
              .set_name("conv1/7x7_s2")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv1/relu_7x7")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool1/3x3_s2")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("pool1/norm1");


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
        (*sub_graph)<< ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv2/3x3_reduce")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3_reduce");


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
        (*sub_graph)<< ConvolutionLayer(
                  3U, 3U, 192U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv2/3x3")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("conv2/norm2")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool2/3x3_s2");


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
        (*sub_graph) << get_inception_node(data_path, "inception_3a", weights_layout, 64, std::make_tuple(96U, 128U), std::make_tuple(16U, 32U), 32U).set_name("inception_3a/concat");

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
        (*sub_graph) << get_inception_node(data_path, "inception_3b", weights_layout, 128, std::make_tuple(128U, 192U), std::make_tuple(32U, 96U), 64U).set_name("inception_3b/concat");
        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool3/3x3_s2");


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
        (*sub_graph) << get_inception_node(data_path, "inception_4a", weights_layout, 192, std::make_tuple(96U, 208U), std::make_tuple(16U, 48U), 64U).set_name("inception_4a/concat");

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
        (*sub_graph) << get_inception_node(data_path, "inception_4b", weights_layout, 160, std::make_tuple(112U, 224U), std::make_tuple(24U, 64U), 64U).set_name("inception_4b/concat");


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
        (*sub_graph) << get_inception_node(data_path, "inception_4c", weights_layout, 128, std::make_tuple(128U, 256U), std::make_tuple(24U, 64U), 64U).set_name("inception_4c/concat");

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
        (*sub_graph) << get_inception_node(data_path, "inception_4d", weights_layout, 112, std::make_tuple(144U, 288U), std::make_tuple(32U, 64U), 64U).set_name("inception_4d/concat");

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
        (*sub_graph) << get_inception_node(data_path, "inception_4e", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_4e/concat");
        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool4/3x3_s2");

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
        (*sub_graph) << get_inception_node(data_path, "inception_5a", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_5a/concat");

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
        (*sub_graph) << get_inception_node(data_path, "inception_5b", weights_layout, 384, std::make_tuple(192U, 384U), std::make_tuple(48U, 128U), 128U).set_name("inception_5b/concat");
        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 7, operation_layout, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL))).set_name("pool5/7x7_s1");



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
        (*sub_graph)<< FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_b.npy"))
              .set_name("loss3/classifier")
              << SoftmaxLayer().set_name("prob")
              << OutputLayer(get_output_accessor(*_common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads = _common_params->threads;
        config.use_tuner   = _common_params->enable_tuner;
        config.tuner_mode  = _common_params->tuner_mode;
        config.tuner_file  = _common_params->tuner_file;
        config.mlgo_file   = _common_params->mlgo_file;



        sub_graph->finalize(_common_params->target, config);


        //arm_compute::graph::Tensor *s_in;
        for(auto &node : sub_graph->graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
            {
                s_in = node->output(0);
            }
        }

        std::cout<<"Partition layer:"<<p<<std::endl;
        std::cout<<"Total layers:"<<Layer+1<<std::endl;
        return true;
    }

    void do_run() override
    {
        // Run graph
        //Ehsan
    	std::thread First(&GraphGooglenetExample::do_run_1,this,0);
    	std::thread Second(&GraphGooglenetExample::do_run_2,this,1);
    	First.join();
    	Second.join();
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
        CPU_SET(0,&set);
        ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");


        std::cout<<"start running graph ...\n";
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
        //CPU_SET(core_id, &set);
        //CPU_SET(1,&set);
        //CPU_SET(0,&set);
        //ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");

        std::cout<<"start running graph ...\n";
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



private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    CommonGraphParams  common_params2;
    Stream			   graph2;
    TensorDescriptor   input_descriptor2;
    bool     		   second=false;

    Stream *sub_graph=&graph;
    CommonGraphParams *_common_params=&common_params;
    int Layer=0;
    int p=0;
    bool			   annotate{false};



    ConcatLayer get_inception_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                   unsigned int a_filt,
                                   std::tuple<unsigned int, unsigned int> b_filters,
                                   std::tuple<unsigned int, unsigned int> c_filters,
                                   unsigned int d_filt)
    {
        std::string total_path = "/cnn_data/googlenet_model/" + param_path + "/" + param_path + "_";
        SubStream   i_a(*sub_graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/1x1")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_1x1");

        SubStream i_b(*sub_graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "3x3_reduce_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "3x3_reduce_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/3x3_reduce")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_3x3_reduce")
            << ConvolutionLayer(
                3U, 3U, std::get<1>(b_filters),
                get_weights_accessor(data_path, total_path + "3x3_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/3x3")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_3x3");

        SubStream i_c(*sub_graph);
        i_c << ConvolutionLayer(
                1U, 1U, std::get<0>(c_filters),
                get_weights_accessor(data_path, total_path + "5x5_reduce_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "5x5_reduce_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/5x5_reduce")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_5x5_reduce")
            << ConvolutionLayer(
                5U, 5U, std::get<1>(c_filters),
                get_weights_accessor(data_path, total_path + "5x5_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "5x5_b.npy"),
                PadStrideInfo(1, 1, 2, 2))
            .set_name(param_path + "/5x5")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_5x5");

        SubStream i_d(*sub_graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, common_params.data_layout, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL))).set_name(param_path + "/pool")
            << ConvolutionLayer(
                1U, 1U, d_filt,
                get_weights_accessor(data_path, total_path + "pool_proj_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "pool_proj_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/pool_proj")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_pool_proj");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};

/** Main program for Googlenet
 *
 * Model is based on:
 *      https://arxiv.org/abs/1409.4842
 *      "Going deeper with convolutions"
 *      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
 *
 * Provenance: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    //ANNOTATE_SETUP;
    return arm_compute::utils::run_example<GraphGooglenetExample>(argc, argv);
}
