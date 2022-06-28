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

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API */
class GraphSqueezenetExample : public Example
{
public:
    GraphSqueezenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "SqueezeNetV1")
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

        annotate=common_params.annotate;
        save_model=common_params.save;


        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
              << ConvolutionLayer(
                  7U, 7U, 96U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_b.npy"),
                  PadStrideInfo(2, 2, 0, 0))
              .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu_conv1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool1")
              << ConvolutionLayer(
                  1U, 1U, 16U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire2/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire2/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire2", weights_layout, 64U, 64U).set_name("fire2/concat");
        graph << ConvolutionLayer(
                  1U, 1U, 16U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire3/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire3/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire3", weights_layout, 64U, 64U).set_name("fire3/concat");
        graph << ConvolutionLayer(
                  1U, 1U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire4/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire4/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire4", weights_layout, 128U, 128U).set_name("fire4/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool4")
              << ConvolutionLayer(
                  1U, 1U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire5/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire5/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire5", weights_layout, 128U, 128U).set_name("fire5/concat");
        graph << ConvolutionLayer(
                  1U, 1U, 48U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire6/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire6/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire6", weights_layout, 192U, 192U).set_name("fire6/concat");
        graph << ConvolutionLayer(
                  1U, 1U, 48U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire7/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire7/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire7", weights_layout, 192U, 192U).set_name("fire7/concat");
        graph << ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire8/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire8/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire8", weights_layout, 256U, 256U).set_name("fire8/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool8")
              << ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("fire9/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("fire9/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire9", weights_layout, 256U, 256U).set_name("fire9/concat");
        graph << ConvolutionLayer(
                  1U, 1U, 1000U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv10")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu_conv10")
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, operation_layout)).set_name("pool10")
              << FlattenLayer().set_name("flatten")
              << SoftmaxLayer().set_name("prob")
              << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads      = common_params.threads;
        config.use_tuner        = common_params.enable_tuner;
        config.tuner_mode       = common_params.tuner_mode;
        config.tuner_file       = common_params.tuner_file;
        config.mlgo_file        = common_params.mlgo_file;
        config.convert_to_uint8 = (common_params.data_type == DataType::QASYMM8);

        graph.finalize(common_params.target, config);

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
        int tt=common_params.n;
        auto tstart=std::chrono::high_resolution_clock::now();
        for(int i=0;i<(tt+1);i++){
        	if(i==1){
        		tstart=std::chrono::high_resolution_clock::now();
        		//std::cout<<tstart.time_since_epoch().count()<<std::endl;
        		in=task=out=0;
        	}
			if(imgs){
					if(image_index>=images_list.size())
							image_index=image_index%images_list.size();
					std::cout<<"\n\ninferencing image: "<<image_index<<":"<<images_list[image_index]<<std::endl;
					//std::unique_ptr<ImageAccessor> im_acc=dynamic_cast<ImageAccessor*>(graph.graph().node(0)->output(0)->accessor());
					im_acc->set_filename(images_list[image_index++]);
			}
			if(annotate)
				graph.run(annotate);
			else
				graph.run();
		}
        auto tfinish=std::chrono::high_resolution_clock::now();
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        double Cost=cost0/tt;
        in=graph.get_input_time()/tt;
        task=graph.get_task_time()/tt;
        out=graph.get_output_time()/tt;
        double tot=in+task+out;
        std::cout<<"Cost:"<<Cost<<std::endl;
        std::cout<<"input_time:"<<in<<"\ntask_time:"<<task<<"\noutput_time:"<<out<<"\ntotal_time:"<<tot<<std::endl;
    }



private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;
    bool			   annotate{false};

    ConcatLayer get_expand_fire_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     unsigned int expand1_filt, unsigned int expand3_filt)
    {
        std::string total_path = "/cnn_data/squeezenet_v1.0_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, expand1_filt,
                get_weights_accessor(data_path, total_path + "expand1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "expand1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/expand1x1")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_expand1x1");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                3U, 3U, expand3_filt,
                get_weights_accessor(data_path, total_path + "expand3x3_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "expand3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/expand3x3")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_expand3x3");

        return ConcatLayer(std::move(i_a), std::move(i_b));
    }
};

/** Main program for Squeezenet v1.0
 *
 * Model is based on:
 *      https://arxiv.org/abs/1602.07360
 *      "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
 *      Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
 *
 * Provenance: https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphSqueezenetExample>(argc, argv);
}
