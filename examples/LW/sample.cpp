#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <bitset>
//#include <android/log.h>
#include <chrono>
#include "rknn_api.h"
#include "rockx.h"

//rknn_tensor_attr outputs_attr[2];
//const int output_index0 = 0;
//const int output_index1 = 1;
rknn_context ctx = 0;



rknn_tensor_attr a;

void print_attr(rknn_tensor_attr attr){
	printf("tensor name:%s\nindex:%d\t n_dims:%d\t",attr.name,attr.index,attr.n_dims);
	printf("Dims: ");
	for(int i=0;i<attr.n_dims;i++){
		printf("%d  ",attr.dims[i]);
	}
	printf("\nelements:%d \t size:%d \t type:%d \t qtype:%d \t scale:%f\n",attr.n_elems,attr.size,(int)attr.type,(int)attr.qnt_type,attr.scale);
}

void print_input(rknn_input input,int n){
	//float* t=(float*)(input.buf);
	//void *t=NULL;
	n=std::min(n,400);
	if(input.type==RKNN_TENSOR_FLOAT32){
		float *t=(float*)(input.buf);
		//int *t=(int*)(input.buf);
		for(int i=0;i<n;i++){
			printf("%-4d:%-9.4f",i,t[i]);
			if(((i+1)%10)==0)
				printf("\n");
		}
	}
	if(input.type==RKNN_TENSOR_FLOAT16){
		//float *t=(float*)(input.buf);
		int16_t *t=(int16_t*)(input.buf);
		for(int i=0;i<n;i++){
			printf("%-4d:%-9.4d",i,t[i]);
			if(((i+1)%10)==0)
				printf("\n");
		}
	}
	if(input.type==RKNN_TENSOR_UINT8){
		auto t=(uint8_t*)(input.buf);
		for(int i=0;i<n;i++){
			printf("%-4d:%-9d",i,t[i] );
			if(((i+1)%10)==0)
				printf("\n");
		}
	}
	if(input.type==RKNN_TENSOR_INT16){
		int16_t *t=(int16_t*)(input.buf);
		for(int i=0;i<n;i++){
			printf("%-4d:%-9d",i,t[i]);
			if(((i+1)%10)==0)
				printf("\n");
		}
	}
	
}

void print_output(rknn_output output,int n){
	n=std::min(n,400);
	float* t=(float*)(output.buf);
	for(int i=0;i<n;i++){
			printf("%-4d:%-12f",i,t[i]);
			if(((i+1)%10)==0){
				printf("\n");
			}
	}
}

void print_image(rockx_image_t image,int n){
	//float* t=(float*)(image.data);
	uint8_t *t=image.data;
	for(int i=0;i<n;i++){
			//printf("%-4d:%-9.1f",i,t[i]);
			printf("%-4d:%-9d",i,t[i]);
			if(((i+1)%10)==0){
				printf("\n");
			}
	}
}

unsigned short float32tofloat16(unsigned int fltInt32){
	//std::cerr<<"Zero:"<<std::bitset<32>(fltInt32)<<std::endl;
	unsigned short fltInt16;
	fltInt16 = (fltInt32 >> 31) << 5;
	//std::cerr<<"First:"<<std::bitset<16>(fltInt16)<<std::endl;
	unsigned short tmp = (fltInt32 >> 23) & 0xff;
	//std::cerr<<"Second:"<<std::bitset<32>(tmp)<<std::endl;
	tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
	//std::cerr<<"Third:"<<std::bitset<16>(tmp)<<std::endl;
	fltInt16 = (fltInt16 | tmp) << 10;
	//std::cerr<<"Forth:"<<std::bitset<16>(tmp)<<std::endl;
	fltInt16 |= (fltInt32 >> 13) & 0x3ff;
	//std::cerr<<"input float32:"<<fltInt32<<'\t'<<"float16:"<<fltInt16<<std::endl;
	return fltInt16;
}

int main(int argc, char *argv[]){
	
	#pragma region
	if (argc<3){
		printf("need model name argument\n");
		return -1;
	}
	char* name=argv[1];

	std::chrono::time_point<std::chrono::high_resolution_clock> t1;
	std::chrono::time_point<std::chrono::high_resolution_clock> t2;
	double t=0;
	std::cerr<<"Reading model...\n";
	//const char* mParamPath="/data/dataset/npu/Alex.rknn";
	std::string mParamPath="/data/dataset/npu/";
	mParamPath.append(name);
	FILE *fp = fopen(mParamPath.c_str(), "rb");
	if(fp == NULL) {
		//LOGE("fopen %s fail!\n", mParamPath);
		printf("fopen %s fail!\n", mParamPath.c_str());
		return -1;
	}
	fseek(fp, 0, SEEK_END);
	int model_len = ftell(fp);
	void *model = malloc(model_len);
	fseek(fp, 0, SEEK_SET);
	if(model_len != fread(model, 1, model_len, fp)) {
		//LOGE("fread %s fail!\n", mParamPath);
		printf("fread %s fail!\n", mParamPath.c_str());
		free(model);
		fclose(fp);
		return -1;
	}
	std::cerr<<"model reading done.\n";

	fclose(fp);

	// RKNN_FLAG_ASYNC_MASK: enable async mode to use NPU efficiently.
	//int ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_PRIOR_MEDIUM|RKNN_FLAG_ASYNC_MASK);
	int ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_PRIOR_MEDIUM);
	std::cerr<<"initialized\n";
	//int ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_COLLECT_PERF_MASK);
	free(model);

	if(ret < 0) {
		//LOGE("rknn_init fail! ret=%d\n", ret);
		printf("rknn_init fail! ret=%d\n", ret);
		return -1;
	}
    	
	rknn_input_output_num io_num;
	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if(ret < 0) {
		printf("rknn_query fail! ret=%d\n",ret);
		return -1;
	}

	std::cerr<<"query in/out nums done.\n";
	rknn_tensor_attr output0_attr;
	output0_attr.index = 0;
	ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output0_attr,
	sizeof(output0_attr));
	if(ret < 0) {
		printf("rknn_query fail! ret=%d\n",ret);
		return -1;
	}
	std::cerr<<"query output attr done.\n";

	rknn_tensor_attr input0_attr;
	input0_attr.index = 0;
	ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input0_attr,
	sizeof(input0_attr));
	if(ret < 0) {
		printf("rknn_query fail! ret=%d\n",ret);
		return -1;
	}
	std::cerr<<"query input attr done.\n";
	

	printf("\n\n*************\n");
	print_attr(input0_attr);
	printf("\n\n*************\n");
	print_attr(output0_attr);

	std::cerr<<"hey:"<<int(input0_attr.dims[0])<<std::endl;
	int H=input0_attr.dims[0];
	int W=input0_attr.dims[1];
	#pragma endregion
	float data[input0_attr.n_elems];
	for(int i=0;i<input0_attr.n_elems;i++){
		data[i]=i;
	}
	/*int C=(int)input0_attr.dims[2];
	float data[H][W][3][1];
	for(int i=0;i<H;i++){
		for(int j=0;j<W;j++){
			for(int k=0;k<3;k++){
				data[i][j][k][0]=20;
			}
		}
	}*/


	int dummy=0;
	bool pass=0;

	rknn_input inputs[1];
	inputs[0].index = 0;
	inputs[0].pass_through = pass;
	inputs[0].fmt = RKNN_TENSOR_NHWC;

	int n=1;
	//Reading image
	/*
	float test_data[input0_attr.n_elems*4];
	float test_data2[input0_attr.n_elems*4];
	float t_data[H][W][3];
	float t_data2[3][H][W];
	unsigned short t_data3[3][H][W];*/
	#pragma region
	if (argc==4){
		const char *img_path = argv[2];
		n=atoi(argv[3]);
		rockx_image_t input_image;
		rockx_image_read(img_path, &input_image, 1);
		std::cout<<"image details:"<<
			"\nheight;"<<input_image.height<<
			"\npixel format:"<<input_image.pixel_format<<
			"\nsize:"<<input_image.size<<
			"\n width:"<<input_image.width<<
			"\ndata[0]:"<<int(input_image.data[0])<<
			"\ndata[1]:"<<int(input_image.data[1])<<
			"\ndata[2]:"<<int(input_image.data[2])<<std::endl;
		
		//std::cout<<"size of image data:"<<sizeof(input_image.data)<<std::endl;

		int input_size=input_image.width*input_image.height*3;

		#pragma region
		//Change BGR image to RGB
		
		float test_data[input_size];
		float test_data2[input_size];
		float t_data[input_image.height][input_image.width][3];
		float t_data2[3][input_image.height][input_image.width];
		unsigned short t_data3[3][input_image.height][input_image.width];
		std::cerr<<"pixel format:"<<int(input_image.pixel_format)<<std::endl;
		return 0;
		//input_image.pixel_format=rockx_pixel_format(1);
		for (int j=0;j<input_size;j+=3){
			if(input_image.pixel_format==2){
				uint8_t t=input_image.data[j];
				input_image.data[j]=input_image.data[j+2];
				input_image.data[j+2]=t;
			}
			test_data[j]=input_image.data[j];
			test_data[j+1]=input_image.data[j+1];
			test_data[j+2]=input_image.data[j+2];
		}
		std::cout<<test_data[223*224*3+218*3+2]<<'\t'<<test_data[223*224*3+219*3+2]<<'\t'<<test_data[223*224*3+220*3+2]<<std::endl;
		std::cout<<test_data[223*224*3+221*3+2]<<'\t'<<test_data[223*224*3+222*3+2]<<'\t'<<test_data[223*224*3+223*3+2]<<std::endl;
		std::cerr<<"up\n";

		//bool preprocess=false;
		//Preprocess:
		if(pass){
			
			for (int j=0;j<input_size;j+=3){
				//input_image.data[j]=input_image.data[j]-122;
				//input_image.data[j+1]=input_image.data[j+1]-116;
				//input_image.data[j+2]=input_image.data[j+2]-104;
				if(input_image.pixel_format==2){
					//std::cerr<<"\n\n\n\n\n\n\n\n\n\n\n";
					test_data[j]=(test_data[j]-123.68)/58.82;
					test_data[j+1]=(test_data[j+1]-116.78)/58.82;
					test_data[j+2]=(test_data[j+2]-103.94)/58.82;
				}
				else{
					test_data[j]=(test_data[j]-103.94)/58.82;
					test_data[j+1]=(test_data[j+1]-116.78)/58.82;
					test_data[j+2]=(test_data[j+2]-123.68)/58.82;
				}
				
			}
			for (int h=0;h<input_image.height;h++){
				for (int w=0;w<input_image.width;w++){
					t_data[h][w][0]=test_data[(h*input_image.width*3)+(w*3)];//-103.94)/58.82;
					t_data[h][w][1]=test_data[(h*input_image.width*3)+(w*3)+1];//-116.78)/58.82;
					t_data[h][w][2]=test_data[(h*input_image.width*3)+(w*3)+2];//-123.68)/58.82;
				}
			}
			std::cerr<<t_data[223][223][0]<<'\t'<<t_data[223][223][1]<<'\t'<<t_data[223][223][2]<<std::endl;
			std::cerr<<t_data[223][221][2]<<'\t'<<t_data[223][222][2]<<'\t'<<t_data[223][223][2]<<std::endl;
			std::cout<<test_data[223*224*3+221*3+2]<<'\t'<<test_data[223*224*3+222*3+2]<<'\t'<<test_data[223*224*3+223*3+2]<<std::endl;
			for (int h=0;h<input_image.height;h++){
				for (int w=0;w<input_image.width;w++){
					for (int c=0;c<3;c++){
						t_data2[c][h][w]=t_data[h][w][c];
					}
				}
			}

			for (int h=0;h<input_image.height;h++){
				for (int w=0;w<input_image.width;w++){
					for (int c=0;c<3;c++){
						t_data3[c][h][w]=float32tofloat16((unsigned int)(t_data[h][w][c]) );
					}
				}
			}

			



			std::cerr<<"Transposed:\n";
			std::cerr<<t_data2[0][0][0]<<'\t'<<t_data2[0][0][1]<<'\t'<<t_data2[0][0][2]<<std::endl;
			std::cerr<<t_data2[0][1][0]<<'\t'<<t_data2[0][1][1]<<'\t'<<t_data2[0][1][2]<<std::endl;
			std::cerr<<t_data2[2][223][221]<<'\t'<<t_data2[2][223][222]<<'\t'<<t_data2[2][223][223]<<std::endl;
			#pragma endregion

		}
		
		print_image(input_image,input_size/1000);

		//Real image
		inputs[0].buf = input_image.data;//test_data;
		inputs[0].size = input_size;//sizeof(test_data)/4;
		inputs[0].type = RKNN_TENSOR_UINT8;
		//inputs[0].buf = t_data3;
		//inputs[0].size =sizeof(t_data3);
		//std::cerr<<"sizeee:"<<sizeof(t_data3);
		//inputs[0].type = RKNN_TENSOR_INT16;
	}
	#pragma endregion
	//Dummy data
	if (argc==3){
		n=atoi(argv[2]);
		dummy=1;
		inputs[0].buf = data;
		inputs[0].size = sizeof(data);
		inputs[0].type = RKNN_TENSOR_FLOAT32;
		//inputs[0].type = RKNN_TENSOR_INT16;
		//inputs[0].type = RKNN_TENSOR_FLOAT16;
	}

	ret = rknn_inputs_set(ctx, 1, inputs);
	
	if(ret < 0) {
		printf("rknn_input_set fail! ret=%d\n", ret);
		return -1;
	}
	
	ret = rknn_run(ctx, NULL);
	
	if(ret < 0) {
		printf("rknn_run fail! ret=%d\n", ret);
		return -1;
	}
	
	rknn_output outputs[1];
	outputs[0].want_float = true;
	outputs[0].is_prealloc = false;

	ret = rknn_outputs_get(ctx, 1, outputs, NULL);
	printf("first test run executed\n");

	printf("\n\n*************\n");
	print_output(outputs[0],output0_attr.n_elems);

	
	#pragma region
	//*****************************************Run with change input without set input
	/*for(int i=0;i<H;i++){
		for(int j=0;j<W;j++){
			for(int k=0;k<3;k++){
				data[i][j][k][0]=40;
			}
		}
	}*/
	/*
	for(int i=0;i<input0_attr.n_elems;i++){
		data[i]=2*i;
	}
	ret = rknn_run(ctx, NULL);
	
	if(ret < 0) {
		printf("rknn_run fail! ret=%d\n", ret);
		return -1;
	}
	ret = rknn_outputs_get(ctx, 1, outputs, NULL);
	printf("\n\n*************\n");*/
	//print_output(outputs[0],output0_attr.n_elems/2);
	//*********************************************
	#pragma endregion
	
	printf("Running %s for %d times...\n",mParamPath.c_str(),n);
	
	double input_time=0;
	double runing_time=0;
	double output_time=0;
	t1=std::chrono::high_resolution_clock::now();
	for(int i=0;i<n;i++){
		//printf("%d test run executed\n",i);
		auto t_start=std::chrono::high_resolution_clock::now();
		ret = rknn_inputs_set(ctx, 1, inputs);
		auto t_input=std::chrono::high_resolution_clock::now();
		ret = rknn_run(ctx, NULL);
		if(ret < 0) {
			printf("rknn_run fail! ret=%d\n", ret);
			return -1;
		}
		auto t_run=std::chrono::high_resolution_clock::now();
		ret = rknn_outputs_get(ctx, 1, outputs, NULL);
		if(ret < 0) {
			printf("rknn_outputs_get fail! ret=%d\n", ret);
			return -1;
		}
		auto t_end=std::chrono::high_resolution_clock::now();
		input_time+=std::chrono::duration_cast<std::chrono::duration<double>>(t_input-t_start).count();
		runing_time+=std::chrono::duration_cast<std::chrono::duration<double>>(t_run-t_input).count();
		output_time+=std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_run).count();

	}
	t2=std::chrono::high_resolution_clock::now();
	auto t3=std::chrono::high_resolution_clock::now();

	double Latency=std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
	//double getout_latency=std::chrono::duration_cast<std::chrono::duration<double>>(t3-t2).count();
	printf("\ninput_time:%f ms\t runing_time:%f ms \t output_time:%f\nLatency:%f\n",1000.0*input_time/n,1000.0*runing_time/n,
	1000.0*output_time/n,1000.0*Latency/n);


	_rknn_perf_run run_time;
	ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &run_time,sizeof(run_time));
	if(ret < 0) {
		printf("rknn_query fail! ret=%d\n",ret);
		return -1;
	}
	printf("run_time:%ld us\n",run_time.run_duration);


	_rknn_sdk_version version;
	ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,sizeof(version));
	if(ret < 0) {
		printf("rknn_query fail! ret=%d\n",ret);
		return -1;
	}
	printf("\n\n*************\n");
	printf("api version:%s \t drive version:%s \n",version.api_version,version.drv_version);
	

	//print_output(outputs[0]);
	printf("\n\n*************\n");
	std::cerr<<input0_attr.n_elems/1000<<std::endl;
	print_input(inputs[0],int(input0_attr.n_elems));
	printf("\n\n*************\n");
	//print_output(outputs[0],output0_attr.n_elems/2);
	
	rknn_perf_detail perf_detail;
	ret = rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail,sizeof(rknn_perf_detail));
	printf("%s", perf_detail.perf_data);

	

	rknn_outputs_release(ctx, 1, outputs);

	rknn_destroy(ctx);

	/*
	outputs_attr[0].index = output_index0;
	ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
	if(ret < 0) {
		//LOGI("rknn_query fail! ret=%d\n", ret);
		std::cerr<<"rknn_query fail!\n";
        return;
    }

	outputs_attr[1].index = output_index1;
	ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
	if(ret < 0) {
		LOGI("rknn_query fail! ret=%d\n", ret);
		return;
	}
 
	rknn_input inputs[1];
	inputs[0].index = input_index;
	inputs[0].buf = inData;
	inputs[0].size = img_width * img_height * img_channels;
	inputs[0].pass_through = false;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	int ret = rknn_inputs_set(ctx, 1, inputs);
	if(ret < 0) {
		LOGE("rknn_input_set fail! ret=%d\n", ret);
		return false;
	}

	ret = rknn_run(ctx, nullptr);
	if(ret < 0) {
		LOGE("rknn_run fail! ret=%d\n", ret);
		return false;
	}

	rknn_output outputs[2];
	#if 0
	outputs[0].want_float = true;
	outputs[0].is_prealloc = true;
	outputs[0].index = output_index0;
	outputs[0].buf = y0;
	outputs[0].size = output_size0;
	outputs[1].want_float = true;
	outputs[1].is_prealloc = true;
	outputs[1].index = output_index1;
	outputs[1].buf = y1;
	outputs[1].size = output_size1;
	#else  // for workround the wrong order issue of output index.
	outputs[0].want_float = true;
	outputs[0].is_prealloc = true;
	outputs[0].index = output_index0;
	outputs[0].buf = y1;
	outputs[0].size = output_size1;
	outputs[1].want_float = true;
	outputs[1].is_prealloc = true;
	outputs[1].index = output_index1;
	outputs[1].buf = y0;
	outputs[1].size = output_size0;
	#endif
	ret = rknn_outputs_get(ctx, 2, outputs, nullptr);
	if(ret < 0) {
		LOGE("rknn_outputs_get fail! ret=%d\n", ret);
		return false;
	}

	rknn_outputs_release(ctx, 2, outputs);
	*/
	return 0;

}
