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
TensorDescriptor input_descriptor2;
TensorDescriptor input_descriptor3;
int Layer=0;
bool second=false;
bool third=false;

/** Example demonstrating how to implement Googlenet's network using the Compute Library's graph API */
class GraphGooglenetExample : public Example
{
public:
    GraphGooglenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GoogleNet1")
    {
    }

    void switch_graph(){
        // Finalize graph
        GraphConfig config;
        if(cluster==0){
        	config.num_threads = common_params.threads2;
        	config.cluster=0;
        }
        else{
        	config.num_threads = common_params.threads;
        	config.cluster=1;
        }
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
        //std::cout << common_params << std::endl;

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
        //common_params2 = consume_common_graph_parameters(common_opts);
        ////common_params2.target=static_cast<arm_compute::graph::Target>(3-int(common_params.target));

        std::string order=common_params.order;
		if(order[0]=='B'){
			common_params.target=static_cast<arm_compute::graph::Target>(1);
			cluster=1;
		}
		else if(order[0]=='L'){
			common_params.target=static_cast<arm_compute::graph::Target>(1);
			cluster=0;
		}
		else if(order[0]=='G'){
			common_params.target=static_cast<arm_compute::graph::Target>(2);
			cluster=2;
			gpu_index=0;
		}


        // Print parameter values
        std::cout << common_params << std::endl;

        p=common_params.partition_point;
        p2=common_params.partition_point2;

        annotate=common_params.annotate;
        save_model=common_params.save;

        //***************************************************************


        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
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
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
        	graph<<OutputLayer(get_output_accessor(common_params, 5));
        	switch_graph();
        	second=true;
        	return true;
        }

        // Layer 2
        graph<< ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv2/3x3_reduce")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3_reduce");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 3
        graph<< ConvolutionLayer(
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
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 4
        graph << get_inception_node(data_path, "inception_3a", weights_layout, 64, std::make_tuple(96U, 128U), std::make_tuple(16U, 32U), 32U).set_name("inception_3a/concat");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 5
        graph << get_inception_node(data_path, "inception_3b", weights_layout, 128, std::make_tuple(128U, 192U), std::make_tuple(32U, 96U), 64U).set_name("inception_3b/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool3/3x3_s2");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 6
        graph << get_inception_node(data_path, "inception_4a", weights_layout, 192, std::make_tuple(96U, 208U), std::make_tuple(16U, 48U), 64U).set_name("inception_4a/concat");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 7
        graph << get_inception_node(data_path, "inception_4b", weights_layout, 160, std::make_tuple(112U, 224U), std::make_tuple(24U, 64U), 64U).set_name("inception_4b/concat");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 8
        graph << get_inception_node(data_path, "inception_4c", weights_layout, 128, std::make_tuple(128U, 256U), std::make_tuple(24U, 64U), 64U).set_name("inception_4c/concat");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 9
        graph << get_inception_node(data_path, "inception_4d", weights_layout, 112, std::make_tuple(144U, 288U), std::make_tuple(32U, 64U), 64U).set_name("inception_4d/concat");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 10
        graph << get_inception_node(data_path, "inception_4e", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_4e/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool4/3x3_s2");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 11
        graph << get_inception_node(data_path, "inception_5a", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_5a/concat");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 12
        graph << get_inception_node(data_path, "inception_5b", weights_layout, 384, std::make_tuple(192U, 384U), std::make_tuple(48U, 128U), 128U).set_name("inception_5b/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 7, operation_layout, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL))).set_name("pool5/7x7_s1");

        Layer++;
		if(Layer==p){
			common_params.labels="transfer";
        	if(order[2]=='G'){
        		common_params.labels="transfer_wait";
        	}
			graph<<OutputLayer(get_output_accessor(common_params, 5));
			switch_graph();
			second=true;
			return true;
		}

        // Layer 13
        graph<< FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_b.npy"))
              .set_name("loss3/classifier")
              << SoftmaxLayer().set_name("prob")
              << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        /*int total_graphs=second+third+1;
        if(total_graphs==3)
        	config.cluster = cluster3;
        if(total_graphs==2)
            config.cluster = cluster2;
        if(total_graphs==1)
            config.cluster = cluster;*/


        config.cluster = cluster;
        if(config.cluster==0){
        	config.num_threads = common_params.threads2;
        }
        else if(config.cluster==1){
        	config.num_threads = common_params.threads;
        }
        //config.num_threads = _common_params->threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;
        //config.cluster=0;


        graph.finalize(common_params.target, config);


        //arm_compute::graph::Tensor *s_in;
        /*for(auto &node : graph.graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
            {
            	if(third)
            		t_in = node->output(0);
            	else if(second)
            		s_in = node->output(0);
            }
        }*/

        std::cout<<"Partition layer:"<<p<<std::endl;
        std::cout<<"Partition layer2:"<<p2<<std::endl;
        //std::cout<<"Total layers:"<<Layer+1<<std::endl;
        return true;
    }

    void do_run() override
    {
        // Run graph
        //Ehsan

        std::cout<<"start running graph ...\n";
        ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
        double in=0;
        double task=0;
        double out=0;
        int tt=(common_params.n);
        auto tstart=std::chrono::high_resolution_clock::now();
        for(int i=0;i<(tt+1);i++){
        	if(i==1){
        		tstart=std::chrono::high_resolution_clock::now();
        		in=task=out=0;
        	}
			if(imgs){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				std::cout<<"\n\ninferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				im_acc->set_filename(images_list[image_index++]);
			}

            graph.run(in,task,out,annotate);
        }
        auto tfinish=std::chrono::high_resolution_clock::now();
	//ANNOTATE_CHANNEL_END(1);
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=cost0/tt;
        in=in/tt;
        task=task/tt;
        out=out/tt;
        double tot=in+task+out;
        std::cout<<"Cost:"<<Cost<<std::endl;
        std::cout<<"input_time:"<<in<<"\ntask_time:"<<task<<"\noutput_time:"<<out<<"\ntotal_time:"<<tot<<std::endl;
	//ANNOTATE_MARKER_STR("20_runs_completed");
    }



private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    //Stream *sub_graph=&graph;
    //CommonGraphParams *_common_params=&common_params;
    //int Layer=0;
    int p=0;
    int p2=0;
    bool			   annotate{false};
    int gpu_index=0;
    int cluster = 0;



    ConcatLayer get_inception_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                   unsigned int a_filt,
                                   std::tuple<unsigned int, unsigned int> b_filters,
                                   std::tuple<unsigned int, unsigned int> c_filters,
                                   unsigned int d_filt)
    {
        std::string total_path = "/cnn_data/googlenet_model/" + param_path + "/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/1x1")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_1x1");

        SubStream i_b(graph);
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

        SubStream i_c(graph);
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

        SubStream i_d(graph);
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






/** Example demonstrating how to implement Googlenet's network using the Compute Library's graph API */
class GraphGooglenetExample2 : public Example
{
public:
    GraphGooglenetExample2()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GoogleNet2")
    {
    }



    void switch_graph(){
    	std::string uu0;
        // Finalize graph
        GraphConfig config;

        if(cluster==0){
        	config.num_threads = common_params.threads2;
        	config.cluster = 0;
        }
        else{
        	config.num_threads = common_params.threads;
        	config.cluster = 1;
        }
        //config.num_threads = common_params.threads;
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
                s_out=node->input(0);
                continue;
            }
        }

        for(auto &node : graph.graph().nodes())
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


        // Create input descriptor
        //const auto        operation_layout = common_params.data_layout;
        ////const TensorShape tensor_shape2     = permute_shape(TensorShape(13U, 13U, 384U, 1U), DataLayout::NCHW, operation_layout);
        const TensorShape tensor_shape3 = s_out->desc().shape;
        const auto        operation_layout = common_params.data_layout;
        input_descriptor3 = TensorDescriptor(tensor_shape3, common_params.data_type).set_layout(operation_layout);


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
        //std::cout << common_params << std::endl;

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
        //common_params2 = consume_common_graph_parameters(common_opts);
        ////common_params2.target=static_cast<arm_compute::graph::Target>(3-int(common_params.target));

        std::string order=common_params.order;
		if(order[2]=='B'){
			common_params.target=static_cast<arm_compute::graph::Target>(1);
			cluster=1;
		}
		else if(order[2]=='L'){
			common_params.target=static_cast<arm_compute::graph::Target>(1);
			cluster=0;
		}
		else if(order[2]=='G'){
			common_params.target=static_cast<arm_compute::graph::Target>(2);
			cluster=2;
			gpu_index=1;
		}

		common_params.image="transfer_wait";
    	if(order[2]=='G'){
    		common_params.labels="transfer";
    	}

        // Print parameter values
        std::cout << common_params << std::endl;


        p=common_params.partition_point;
        p2=common_params.partition_point2;

        annotate=common_params.annotate;
        save_model=common_params.save;

        //***************************************************************
        graph << common_params.target
        		<< common_params.fast_math_hint
        		<< InputLayer(input_descriptor2, get_input_accessor(common_params));


        if(Layer==0){
			graph << common_params.target
				  << common_params.fast_math_hint
				  << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
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
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }


        // Layer 2
        if(Layer==1){
			graph<< ConvolutionLayer(
					  1U, 1U, 64U,
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_w.npy", weights_layout),
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_b.npy"),
					  PadStrideInfo(1, 1, 0, 0))
				  .set_name("conv2/3x3_reduce")
				  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3_reduce");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 3
        if(Layer==2){
			graph<< ConvolutionLayer(
					  3U, 3U, 192U,
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_w.npy", weights_layout),
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_b.npy"),
					  PadStrideInfo(1, 1, 1, 1))
				  .set_name("conv2/3x3")
				  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3")
				  << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("conv2/norm2")
				  << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool2/3x3_s2");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 4
        if(Layer==3){
			graph << get_inception_node(data_path, "inception_3a", weights_layout, 64, std::make_tuple(96U, 128U), std::make_tuple(16U, 32U), 32U).set_name("inception_3a/concat");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 5
        if(Layer==4){
			graph << get_inception_node(data_path, "inception_3b", weights_layout, 128, std::make_tuple(128U, 192U), std::make_tuple(32U, 96U), 64U).set_name("inception_3b/concat");
			graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool3/3x3_s2");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 6
        if(Layer==5){
			graph << get_inception_node(data_path, "inception_4a", weights_layout, 192, std::make_tuple(96U, 208U), std::make_tuple(16U, 48U), 64U).set_name("inception_4a/concat");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 7
        if(Layer==6){
			graph << get_inception_node(data_path, "inception_4b", weights_layout, 160, std::make_tuple(112U, 224U), std::make_tuple(24U, 64U), 64U).set_name("inception_4b/concat");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 8
        if(Layer==7){
			graph << get_inception_node(data_path, "inception_4c", weights_layout, 128, std::make_tuple(128U, 256U), std::make_tuple(24U, 64U), 64U).set_name("inception_4c/concat");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 9
        if(Layer==8){
			graph << get_inception_node(data_path, "inception_4d", weights_layout, 112, std::make_tuple(144U, 288U), std::make_tuple(32U, 64U), 64U).set_name("inception_4d/concat");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 10
        if(Layer==9){
			graph << get_inception_node(data_path, "inception_4e", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_4e/concat");
			graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool4/3x3_s2");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 11
        if(Layer==10){
			graph << get_inception_node(data_path, "inception_5a", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_5a/concat");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 12
        if(Layer==11){
			graph << get_inception_node(data_path, "inception_5b", weights_layout, 384, std::make_tuple(192U, 384U), std::make_tuple(48U, 128U), 128U).set_name("inception_5b/concat");
			graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 7, operation_layout, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL))).set_name("pool5/7x7_s1");

			Layer++;
			if(Layer==p2){
				common_params.labels="transfer";
	        	if(order[4]=='G'){
	        		common_params.labels="transfer_wait";
	        	}
				graph<<OutputLayer(get_output_accessor(common_params, 5));
				switch_graph();
				third=true;
				return true;
			}
        }

        // Layer 13
        if(Layer==12){
			graph<< FullyConnectedLayer(
					  1000U,
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_w.npy", weights_layout),
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_b.npy"))
				  .set_name("loss3/classifier")
				  << SoftmaxLayer().set_name("prob")
				  << OutputLayer(get_output_accessor(common_params, 5));
        }


        // Finalize graph
        GraphConfig config;
        /*int total_graphs=second+third+1;
        if(total_graphs==3)
        	config.cluster = cluster3;
        if(total_graphs==2)
            config.cluster = cluster2;
        if(total_graphs==1)
            config.cluster = cluster;*/

        config.cluster = cluster;
        if(config.cluster==0){
        	config.num_threads = common_params.threads2;
        }
        else if(config.cluster==1){
        	config.num_threads = common_params.threads;
        }
        //config.num_threads = _common_params->threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;
        //config.cluster=0;


        graph.finalize(common_params.target, config);


        //arm_compute::graph::Tensor *s_in;
        for(auto &node : graph.graph().nodes())
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
        std::cout<<"Total layers:"<<Layer+1<<std::endl;
        return true;
    }

    void do_run() override
    {
        // Run graph
        //Ehsan
        std::cout<<"start running graph ...\n";
        ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
        double in=0;
        double task=0;
        double out=0;
        int tt=(common_params.n);
        auto tstart=std::chrono::high_resolution_clock::now();
        for(int i=0;i<(tt+1);i++){
        	if(i==1){
        		tstart=std::chrono::high_resolution_clock::now();
        		in=task=out=0;
        	}
			if(imgs){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				std::cout<<"\n\ninferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				im_acc->set_filename(images_list[image_index++]);
			}

            graph.run(in,task,out,annotate);

        }
        auto tfinish=std::chrono::high_resolution_clock::now();
	//ANNOTATE_CHANNEL_END(1);
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=cost0/tt;
        in=in/tt;
        task=task/tt;
        out=out/tt;
        double tot=in+task+out;
        std::cout<<"Cost:"<<Cost<<std::endl;
        std::cout<<"input2_time:"<<in<<"\ntask2_time:"<<task<<"\noutput2_time:"<<out<<"\ntotal2_time:"<<tot<<std::endl;
	//ANNOTATE_MARKER_STR("20_runs_completed");
    }



private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    //Stream *sub_graph=&graph;
    //CommonGraphParams *_common_params=&common_params;
    //int Layer=0;
    int p=0;
    int p2=0;
    bool			   annotate{false};
    int gpu_index=0;
    int cluster = 0;



    ConcatLayer get_inception_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                   unsigned int a_filt,
                                   std::tuple<unsigned int, unsigned int> b_filters,
                                   std::tuple<unsigned int, unsigned int> c_filters,
                                   unsigned int d_filt)
    {
        std::string total_path = "/cnn_data/googlenet_model/" + param_path + "/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/1x1")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_1x1");

        SubStream i_b(graph);
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

        SubStream i_c(graph);
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

        SubStream i_d(graph);
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








/** Example demonstrating how to implement Googlenet's network using the Compute Library's graph API */
class GraphGooglenetExample3 : public Example
{
public:
    GraphGooglenetExample3()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GoogleNet3")
    {
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
        //common_params2 = consume_common_graph_parameters(common_opts);
        ////common_params2.target=static_cast<arm_compute::graph::Target>(3-int(common_params.target));

        std::string order=common_params.order;
		if(order[4]=='B'){
			common_params.target=static_cast<arm_compute::graph::Target>(1);
			cluster=1;
		}
		else if(order[4]=='L'){
			common_params.target=static_cast<arm_compute::graph::Target>(1);
			cluster=0;
		}
		else if(order[4]=='G'){
			common_params.target=static_cast<arm_compute::graph::Target>(2);
			cluster=2;
			gpu_index=2;
		}

		common_params.image="transfer_wait";
    	if(order[4]=='G'){
    		common_params.labels="transfer";
    	}

		// Print parameter values
		std::cout << common_params << std::endl;


        p=common_params.partition_point;
        p2=common_params.partition_point2;


        annotate=common_params.annotate;
        save_model=common_params.save;

        //***************************************************************

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor3, get_input_accessor(common_params));

        if(Layer==0){
			graph << common_params.target
				  << common_params.fast_math_hint
				  << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
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
        }

        // Layer 2
        if(Layer==1){
			graph<< ConvolutionLayer(
					  1U, 1U, 64U,
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_w.npy", weights_layout),
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_b.npy"),
					  PadStrideInfo(1, 1, 0, 0))
				  .set_name("conv2/3x3_reduce")
				  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3_reduce");

			Layer++;
        }

        // Layer 3
        if(Layer==2){
			graph<< ConvolutionLayer(
					  3U, 3U, 192U,
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_w.npy", weights_layout),
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_b.npy"),
					  PadStrideInfo(1, 1, 1, 1))
				  .set_name("conv2/3x3")
				  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3")
				  << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("conv2/norm2")
				  << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool2/3x3_s2");

			Layer++;
        }

        // Layer 4
        if(Layer==3){
			graph << get_inception_node(data_path, "inception_3a", weights_layout, 64, std::make_tuple(96U, 128U), std::make_tuple(16U, 32U), 32U).set_name("inception_3a/concat");

			Layer++;
        }

        // Layer 5
        if(Layer==4){
			graph << get_inception_node(data_path, "inception_3b", weights_layout, 128, std::make_tuple(128U, 192U), std::make_tuple(32U, 96U), 64U).set_name("inception_3b/concat");
			graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool3/3x3_s2");

			Layer++;
        }

        // Layer 6
        if(Layer==5){
			graph << get_inception_node(data_path, "inception_4a", weights_layout, 192, std::make_tuple(96U, 208U), std::make_tuple(16U, 48U), 64U).set_name("inception_4a/concat");
			Layer++;
        }

        // Layer 7
        if(Layer==6){
			graph << get_inception_node(data_path, "inception_4b", weights_layout, 160, std::make_tuple(112U, 224U), std::make_tuple(24U, 64U), 64U).set_name("inception_4b/concat");

			Layer++;
        }

        // Layer 8
        if(Layer==7){
			graph << get_inception_node(data_path, "inception_4c", weights_layout, 128, std::make_tuple(128U, 256U), std::make_tuple(24U, 64U), 64U).set_name("inception_4c/concat");

			Layer++;
        }

        // Layer 9
        if(Layer==8){
			graph << get_inception_node(data_path, "inception_4d", weights_layout, 112, std::make_tuple(144U, 288U), std::make_tuple(32U, 64U), 64U).set_name("inception_4d/concat");

			Layer++;
        }

        // Layer 10
        if(Layer==9){
			graph << get_inception_node(data_path, "inception_4e", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_4e/concat");
			graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool4/3x3_s2");

			Layer++;
        }

        // Layer 11
        if(Layer==10){
			graph << get_inception_node(data_path, "inception_5a", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_5a/concat");

			Layer++;
        }

        // Layer 12
        if(Layer==11){
			graph << get_inception_node(data_path, "inception_5b", weights_layout, 384, std::make_tuple(192U, 384U), std::make_tuple(48U, 128U), 128U).set_name("inception_5b/concat");
			graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 7, operation_layout, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL))).set_name("pool5/7x7_s1");

			Layer++;
        }

        // Layer 13
        if(Layer==12){
			graph<< FullyConnectedLayer(
					  1000U,
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_w.npy", weights_layout),
					  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_b.npy"))
				  .set_name("loss3/classifier")
				  << SoftmaxLayer().set_name("prob")
				  << OutputLayer(get_output_accessor(common_params, 5));
        }

        // Finalize graph
        GraphConfig config;
        /*int total_graphs=second+third+1;
        if(total_graphs==3)
        	config.cluster = cluster3;
        if(total_graphs==2)
            config.cluster = cluster2;
        if(total_graphs==1)
            config.cluster = cluster;*/

        config.cluster = cluster;
        if(config.cluster==0){
        	config.num_threads = common_params.threads2;
        }
        else if(config.cluster==1){
        	config.num_threads = common_params.threads;
        }
        //config.num_threads = _common_params->threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;
        //config.cluster=0;


        graph.finalize(common_params.target, config);


        //arm_compute::graph::Tensor *s_in;
        for(auto &node : graph.graph().nodes())
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
        std::cout<<"Total layers:"<<Layer+1<<std::endl;
        return true;
    }
    void do_run() override
    {
        // Run graph
        //Ehsan
        std::cout<<"start running graph ...\n";
        ImageAccessor *im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
        double in=0;
        double task=0;
        double out=0;
        int tt=(common_params.n);
        auto tstart=std::chrono::high_resolution_clock::now();
        for(int i=0;i<(tt+1);i++){
        	if(i==1){
        		tstart=std::chrono::high_resolution_clock::now();
        		in=task=out=0;
        	}
			if(imgs){
				if(image_index>=images_list.size())
						image_index=image_index%images_list.size();
				std::cout<<"\n\ninferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
				//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
				im_acc->set_filename(images_list[image_index++]);
			}

            graph.run(in,task,out,annotate);
        }
        auto tfinish=std::chrono::high_resolution_clock::now();
	//ANNOTATE_CHANNEL_END(1);
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=cost0/tt;
        in=in/tt;
        task=task/tt;
        out=out/tt;
        double tot=in+task+out;
        std::cout<<"Cost:"<<Cost<<std::endl;
        std::cout<<"input3_time:"<<in<<"\ntask3_time:"<<task<<"\noutput3_time:"<<out<<"\ntotal3_time:"<<tot<<std::endl;
	//ANNOTATE_MARKER_STR("20_runs_completed");
    }



private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    //Stream *sub_graph=&graph;
    //CommonGraphParams *_common_params=&common_params;
    //int Layer=0;
    int p=0;
    int p2=0;
    bool			   annotate{false};
    int gpu_index=0;
    int cluster = 0;



    ConcatLayer get_inception_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                   unsigned int a_filt,
                                   std::tuple<unsigned int, unsigned int> b_filters,
                                   std::tuple<unsigned int, unsigned int> c_filters,
                                   unsigned int d_filt)
    {
        std::string total_path = "/cnn_data/googlenet_model/" + param_path + "/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/1x1")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_1x1");

        SubStream i_b(graph);
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

        SubStream i_c(graph);
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

        SubStream i_d(graph);
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
	bool ret=0;
    ret=ret || arm_compute::utils::run_example<GraphGooglenetExample>(argc, argv);

    if(second)
    	ret=ret || arm_compute::utils::run_example<GraphGooglenetExample2>(argc, argv);

    if(third)
    	ret=ret || arm_compute::utils::run_example<GraphGooglenetExample3>(argc, argv);

    return ret;
}
