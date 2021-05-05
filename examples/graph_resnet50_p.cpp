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
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "ResNetV1_50"), common_params2(), graph2(1,"ResNetV1_50_2")
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

        std::cout<<"Layer:"<<Layer<<std::endl;
        std::cout<<"Layer shape:"<<tensor_shape2<<std::endl;
        std::cout<<"press to continue ...";
        std::string tt;
        std::cin>>tt;


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
        std::cout << common_params << std::endl;

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

        //common_params.target=arm_compute::graph::Target ::CL;
        //common_params2.target=arm_compute::graph::Target ::NEON;
        //std::cout<<int(common_params.target)<<std::endl<<std::endl;
        common_params2.target=static_cast<arm_compute::graph::Target>(3-int(common_params.target));
        std::cout<<"\nFirst sub graph device:"<<common_params.target<<" Second sub graph device:"<<common_params2.target
        		<<"\n\n";

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
        //***************************************************************



        (*sub_graph) << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false /* Do not convert to BGR */))
              << ConvolutionLayer(
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

        std::string test;
        Layer++;
		if(Layer==p){
			//std::cout<<"first layer:"<<Layer<<','<<p<<std::endl;
			//std::cin>>test;
			common_params.labels="transfer";
			(*sub_graph)<<OutputLayer(get_output_accessor(*_common_params, 5));
			sub_graph=&graph2;
			_common_params=&common_params2;
			switch_graph();
			(*sub_graph)<<InputLayer(input_descriptor2, get_input_accessor(*_common_params));
			second=true;
		}



        add_residual_block(data_path, "block1", weights_layout, 64, 3, 2);
        add_residual_block(data_path, "block2", weights_layout, 128, 4, 2);
        add_residual_block(data_path, "block3", weights_layout, 256, 6, 2);
        add_residual_block(data_path, "block4", weights_layout, 512, 3, 1);

        (*sub_graph) << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, operation_layout)).set_name("pool5")
              << ConvolutionLayer(
                  1U, 1U, 1000U,
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_biases.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("logits/convolution")
              << FlattenLayer().set_name("predictions/Reshape")
              << SoftmaxLayer().set_name("predictions/Softmax")
              << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads      = _common_params->threads;
        config.use_tuner        = _common_params->enable_tuner;
        config.tuner_mode       = _common_params->tuner_mode;
        config.tuner_file       = _common_params->tuner_file;
        config.mlgo_file        = _common_params->mlgo_file;
        config.convert_to_uint8 = (_common_params->data_type == DataType::QASYMM8);

        sub_graph->finalize(_common_params->target, config);


        //arm_compute::graph::Tensor *s_in;
        for(auto &node : sub_graph->graph().nodes())
        {
            if(node != nullptr && node->type() == arm_compute::graph::NodeType::Input)
            {
                s_in = node->output(0);
            }
        }


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

        double in2=0;
        double task2=0;
        double out2=0;


        int tt=1;
        auto tstart=std::chrono::high_resolution_clock::now();
        for(int i=0;i<tt;i++){
        	if(imgs){
		        if(image_index>=images_list.size())
		                image_index=image_index%images_list.size();
		        std::cout<<"\n\ninferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
		        //std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
		        im_acc->set_filename(images_list[image_index++]);
        	}
            graph.run(in,task,out,1);
            if(second)
            	graph2.run(in2,task2,out2);

        }
        auto tfinish=std::chrono::high_resolution_clock::now();
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=cost0/tt;
        in=in/tt;
        task=task/tt;
        out=out/tt;
        double tot=in+task+out;
        std::cout<<"Cost:"<<Cost<<std::endl;
        std::cout<<"input_time:"<<in<<"\ntask_time:"<<task<<"\noutput_time:"<<out<<"\ntotal_time:"<<tot<<std::endl;
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

    Stream 			   *sub_graph=&graph;
    CommonGraphParams  *_common_params=&common_params;
    int Layer=0;
    int p=0;


    void add_residual_block(const std::string &data_path, const std::string &name, DataLayout weights_layout,
                            unsigned int base_depth, unsigned int num_units, unsigned int stride)
    {
        for(unsigned int i = 0; i < num_units; ++i)
        {
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

            SubStream right((*sub_graph));
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
                SubStream left((*sub_graph));
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

                (*sub_graph) << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(unit_name + "add");
            }
            else if(middle_stride > 1)
            {
                SubStream left((*sub_graph));
                left << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 1, common_params.data_layout, PadStrideInfo(middle_stride, middle_stride, 0, 0), true)).set_name(unit_name + "shortcut/MaxPool");

                (*sub_graph) << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(unit_name + "add");
            }
            else
            {
                SubStream left((*sub_graph));
                (*sub_graph) << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(unit_name + "add");
            }

            (*sub_graph) << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");
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
