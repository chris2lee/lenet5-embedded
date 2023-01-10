#include "improvedlenet5.h"
#include "params.h"
// #include "lenet5params.h"
#include <stdio.h>
/*
	各层参数
	net=torch.nn.Sequential(
	nn.Conv2d(3,6,kernel_size=5,padding=2),nn.ReLU(),
	nn.MaxPool2d(kernel_size=2,stride=2),
	nn.Conv2d(6,16,kernel_size=5),nn.ReLU(),
	nn.MaxPool2d(kernel_size=2,stride=2),nn.Flatten(),
	nn.Linear(576,120),nn.ReLU(),
	nn.Linear(120,84),nn.ReLU(),
	nn.Linear(84,2)
	各层输出
	Conv2d output shape:	 torch.Size([1, 6, 32, 32])
	ReLU output shape:		 torch.Size([1, 6, 32, 32])
	MaxPool2d output shape:	 torch.Size([1, 6, 16, 16])
	Conv2d output shape:	 torch.Size([1, 16, 12, 12])
	ReLU output shape:		 torch.Size([1, 16, 12, 12])
	MaxPool2d output shape:	 torch.Size([1, 16, 6, 6])
	Flatten output shape:	 torch.Size([1, 576])
	Linear output shape:	 torch.Size([1, 120])
	ReLU output shape:		 torch.Size([1, 120])
	Linear output shape:	 torch.Size([1, 84])
	ReLU output shape:		 torch.Size([1, 84])
	Linear output shape:	 torch.Size([1, 2])
*/

extern float conv1Weight[6][1][5][5];
extern float conv2Weight[16][6][5][5];
extern float fullConnect1Weight[120][256];
extern float fullConnect2Weight[64][120];
extern float fullConnect3Weight[10][64];

extern float fullConnect1Bias[120];
extern float fullConnect2Bias[64];
extern float fullConnect3Bias[10];

extern float conv1Bias[6];
extern float conv2Bias[16];

static void mallocForppptr(float*** imageSpace,imageSize spaceSize) 
{
	for (int k = 0; k < spaceSize.dimsc; k++)
	{
		imageSpace[k] = (float**)malloc(spaceSize.rowsc * sizeof(float*));
		while(imageSpace[k] == NULL){
			printf("imageSpace[i] malloc failed\n");
			imageSpace[k] = (float**)malloc(spaceSize.rowsc * sizeof(float*));
		}	
	}

	for (int i = 0; i < spaceSize.dimsc; i++)
		for(int j=0;j<spaceSize.rowsc;j++){
			imageSpace[i][j] = (float*)malloc(spaceSize.colsc * sizeof(float));
			while(imageSpace == NULL){
				printf("imageSpace[j] malloc failed\n");
				imageSpace[i][j] = (float*)malloc(spaceSize.colsc * sizeof(float));
			}
		}
}
// static void mallocForppptr(float*** imageSpace,imageSize spaceSize) //我改得
// {	
// 	for (int k = 0; k < spaceSize.dimsc; k++)
// 	{
// 		imageSpace[k] = (float**)malloc(spaceSize.rowsc * sizeof(float*));
// 		for(int i=0;i<spaceSize.rowsc;i++){
// 			imageSpace[k][i] = (float*)malloc(spaceSize.colsc * sizeof(float));
// 		}
// 	}
// }

static void freeForppptr(float*** imageSpace,imageSize spaceSize)  //imageSpace一个值  spaceSize一个数组
{
	// Optimized your memory allocation code
	printf("freeForppptr start\n");
	for (int k = 0; k < spaceSize.dimsc; k++)
	{
		for (int i = 0; i < spaceSize.rowsc; i++)
		{
			free((void*)imageSpace[k][i]);	
		}
		free((void**)imageSpace[k]);
		// free((void*)imageSpace[k]);
	}
	printf("freeForppptr end\n");
}

static void freeKernel(float**** imageSpace,imageSize spaceSize)  //imageSpace一个值  spaceSize一个数组
{	
	// Optimized your memory allocation code
	printf("freeKernel start\n");
	for(int h=0;h<spaceSize.numsc;h++){
		for (int k = 0; k < spaceSize.dimsc; k++){
			for (int i = 0; i < spaceSize.rowsc; i++){
				free((void*)imageSpace[h][k][i]);	
			}
			free((void**)imageSpace[h][k]);
			// free((void*)imageSpace[k]);
		}
		free((void***)imageSpace[h]);
	}
	printf("freeKernel end\n");
}

/*
	@params
		input: the input matrix of lenet5 
		inputSize: a imageSize of the input matrix (type float****)  
	@retval 
		res: the classify result, using the number of 少要编写这么几个处node in ouput layer
*/
int lenet5Improved(float*** input,imageSize inputSize)
{
	
	imageSize OutSize1 = { 1,6,24,24 };   //输出
	// imageSize OutSize1 = { 1,3,32,32 };
	float*** Out1 = (float***)malloc(OutSize1.dimsc*sizeof(float**)); 
	// float*** Out1 = (float***)malloc(OutSize1.dimsc*sizeof(float**)); 
	mallocForppptr(Out1,OutSize1);     //申请内存空间,out1一个值  outsize1一个向量
	
	imageSize kernelSize1 = { 6,1,5,5 };  
	// imageSize kernelSize1 = { 6,1,5,5 };
    // 赋权重
	float**** kernel1 = (float****)malloc(kernelSize1.numsc*sizeof(float***));
	for(int n=0;n<kernelSize1.numsc;n++){                                     
		// kernel1[n] = (float***)malloc(kernelSize1.rowsc*sizeof(float**));
		kernel1[n] = (float***)malloc(kernelSize1.dimsc*sizeof(float**));
		for(int d=0;d<kernelSize1.dimsc;d++){
			kernel1[n][d] = (float**)malloc(kernelSize1.rowsc*sizeof(float*));
			for(int i=0;i<kernelSize1.rowsc;i++){
				kernel1[n][d][i] = (float*)malloc(kernelSize1.colsc*sizeof(float));
				for(int j = 0;j<kernelSize1.colsc;j++){
					kernel1[n][d][i][j] =  conv1Weight[n][d][i][j];		                       
				}
			}
		}		
	}
	
	float* bias1 = (float*)malloc(OutSize1.dimsc*sizeof(float)); 
	for(int i=0;i<OutSize1.dimsc;i++){
		bias1[i] = conv1Bias[i];
	} 
	//设置padding和步长
	int padding1 = 0;
	int step1 = 1;
	
	//第一层 卷积处理
	cnnOperationConvolution(input, inputSize,Out1, OutSize1, kernel1, kernelSize1,padding1,step1,bias1);
	// cnnOperationConvolution(input, inputSize,Out1, OutSize1, kernel1, kernelSize1,padding1,step1,bias1); //1*6*32*32
	freeForppptr(input,inputSize);
	free((float*)bias1);
	freeKernel(kernel1,kernelSize1);
	//激活函数
	cnnOperationActivation((float***)Out1, OutSize1, 0);      

	//第二层 池化处理
	imageSize OutSize2 = { 1,6,12,12 };
	float*** Out2 = (float***)malloc(OutSize2.dimsc*sizeof(float**));
	mallocForppptr(Out2,OutSize2);
	imageSize kernelSize2 = { 1,1,2,2 };
	//float**** kernel2 = (float****)malloc(kernelSize2.numsc*sizeof(float***));
	
	int padding2 = 0;
	int step2 = 2;
	cnnOperationPooling(Out1,OutSize1,Out2,OutSize2, kernelSize2, padding2,step2);  //1*6*16*16
	freeForppptr(Out1,OutSize1);

	//第三层 卷积处理
	imageSize OutSize3 = { 1,16,8,8 };
	float*** Out3 = (float***)malloc(OutSize3.dimsc*sizeof(float**));     
	mallocForppptr(Out3, OutSize3);
	imageSize kernelSize3 = { 16,6,5,5 };
	float**** kernel3 = (float****)malloc(kernelSize3.numsc*sizeof(float***));
	for(int n=0;n<kernelSize3.numsc;n++){
		// kernel3[n] = (float***)malloc(kernelSize3.rowsc*sizeof(float**));
		kernel3[n] = (float***)malloc(kernelSize3.dimsc*sizeof(float**));
		for(int d=0;d<kernelSize3.dimsc;d++){
			kernel3[n][d] = (float**)malloc(kernelSize3.rowsc*sizeof(float*));
			for(int i=0;i<kernelSize3.rowsc;i++){
				kernel3[n][d][i] = (float*)malloc(kernelSize3.colsc*sizeof(float));
				for(int j = 0;j<kernelSize3.colsc;j++){
					kernel3[n][d][i][j] =  conv2Weight[n][d][i][j];			
				}
			}
		}		
	}

	float* bias2 = (float*)malloc(OutSize2.dimsc*sizeof(float)); 
	for(int i=0;i<OutSize2.dimsc;i++){
		bias2[i] = conv2Bias[i];
	} 

	int padding3 = 0;
	int step3 = 1;
	cnnOperationConvolution(Out2, OutSize2, Out3, OutSize3, kernel3, kernelSize3, padding3, step3, bias2);   //1*6*12*12
	freeForppptr(Out2,OutSize2);
	freeKernel(kernel3,kernelSize3);
	free((float*)bias2);

	// float**** kernel31 = (float****)malloc(kernelSize3.numsc*sizeof(float***));
	// for(int n=0;n<OutSize2.numsc;n++){
	// // kernel3[n] = (float***)malloc(kernelSize3.rowsc*sizeof(float**));
	// 	kernel31[n] = (float***)malloc(OutSize2.dimsc*sizeof(float**));
	// 	for(int d=0;d<OutSize2.dimsc;d++){
	// 		kernel31[n][d] = (float**)malloc(OutSize2.rowsc*sizeof(float*));
	// 		for(int i=0;i<OutSize2.rowsc;i++){
	// 			kernel31[n][d][i] = (float*)malloc(OutSize2.colsc*sizeof(float));
	// 			for(int j = 0;j<OutSize2.colsc;j++){
	// 				kernel31[n][d][i][j] +=  bias2[i];			
	// 			}
	// 		}
	// 	}		
	// }
	
	//激活函数
	cnnOperationActivation((float***)Out3, OutSize3, 0);
	//第四层 池化处理
	imageSize OutSize4 = { 16,1,4,4 };
	// float*** Out4 = (float***)malloc(OutSize4.dimsc*sizeof(float**)); 
	float*** Out4 = (float***)malloc(OutSize4.dimsc*sizeof(float**));              

	mallocForppptr(Out4, OutSize4);
	imageSize kernelSize4 = {1,1,2,2};
	// float**** kernel4 = NULL;
	int padding4 = 0;
	int step4 = 2;

	cnnOperationPooling(Out3, OutSize3, Out4, OutSize4,kernelSize4, padding4,step4);         //16*1*6*6
	freeForppptr(Out3,OutSize3);
	// freeKernel(kernel4,kernelSize4);
	//第五层 扁平化

	imageSize OutSize5 = { 1,1,1,256};
	float*** Out5 = (float***)malloc(OutSize5.dimsc*sizeof(float**)); 
	mallocForppptr(Out5, OutSize5);
	cnnOperationFlatten(Out4, OutSize4, Out5, OutSize5);
	freeForppptr(Out4,OutSize4);
	//第六层 全连接
	imageSize OutSize6 = { 1,1,1,120 };
	float*** Out6 = (float***)malloc(OutSize6.dimsc*sizeof(float**)); 
	mallocForppptr(Out6, OutSize6);
	float** weight6 = (float**)malloc(OutSize6.colsc*sizeof(float*)); 
	for(int i=0;i<OutSize6.colsc;i++){
		weight6[i] = (float*)malloc(OutSize5.colsc*sizeof(float));
		for(int j=0;j<OutSize5.colsc;j++){
			weight6[i][j] = fullConnect1Weight[i][j];
		}	
	}
	float* bias6 = (float*)malloc(OutSize6.colsc*sizeof(float)); 
	for(int i=0;i<OutSize6.colsc;i++){
		bias6[i] = fullConnect1Bias[i];
	} 
	cnnOperationLinear(Out5, OutSize5, Out6, OutSize6,weight6,bias6);
	freeForppptr(Out5,OutSize5);
	free((float*)bias6);
	//激活函数
	cnnOperationActivation((float***)Out6, OutSize6, 0);
	//第七层 全连接

	imageSize OutSize7 = { 1,1,1,64 };
	float*** Out7 = (float***)malloc(OutSize7.dimsc*sizeof(float**)); 
	mallocForppptr(Out7, OutSize7);
	float** weight7 = (float**)malloc(OutSize7.colsc*sizeof(float*)); 
	for(int i=0;i<OutSize7.colsc;i++){
		weight7[i] = (float*)malloc(OutSize6.colsc*sizeof(float));
		for(int j=0;j<OutSize6.colsc;j++)
			weight7[i][j] = fullConnect2Weight[i][j];
	}
	float* bias7 = (float*)malloc(OutSize7.colsc*sizeof(float)); 
	for(int i=0;i<OutSize7.colsc;i++){
		bias7[i] = fullConnect2Bias[i];
	} 

	cnnOperationLinear(Out6, OutSize6, Out7, OutSize7,weight7,bias7);
	freeForppptr(Out6,OutSize6);
	free((float*)bias7);
	//激活函数
	cnnOperationActivation((float***)Out7, OutSize7, 0);
	//第八层 全连接
	imageSize OutSize8 = { 1,1,1,10 };
	float*** Out8 = (float***)malloc(OutSize8.dimsc*sizeof(float**)); 
	mallocForppptr(Out8, OutSize8);
	float** weight8 = (float**)malloc(OutSize8.colsc*sizeof(float*)); 
	for(int i=0;i<OutSize8.colsc;i++){
		weight8[i] = (float*)malloc(OutSize7.colsc*sizeof(float));
		for(int j=0;j<OutSize7.colsc;j++){
			weight8[i][j] = fullConnect3Weight[i][j];
		}
	}
	float* bias8 = (float*)malloc(OutSize8.colsc*sizeof(float)); 
	for(int i=0;i<OutSize8.colsc;i++){
		bias8[i] = fullConnect3Bias[i];
	} 

	cnnOperationLinear(Out7, OutSize7, Out8, OutSize8, weight8,bias8);
	freeForppptr(Out7,OutSize7);
	free((float*)bias8);
	cnnOperationActivation((float***)Out8, OutSize8, 0);
	
	// int *resptr = (int*)malloc(sizeof(int));

	// matOperationMaxIt(Out8[0],OutSize8,resptr);
	// printf("resptr=%d\n",*resptr);
	
	printf("out8[0][0][0]=%.3lf out8[0][0][1]=%.3lf out8[0][0][2]=%.3lf out8[0][0][3]=%.3lf out8[0][0][4]=%.3lf\n out8[0][0][5]=%.3lf out8[0][0][6]=%.3lf out8[0][0][7]=%.3lf out8[0][0][8]=%.3lf out8[0][0][9]=%.3lf\n ",
	        Out8[0][0][0],      Out8[0][0][1],      Out8[0][0][2],      Out8[0][0][3],      Out8[0][0][4],        Out8[0][0][5],      Out8[0][0][6],      Out8[0][0][7],      Out8[0][0][8],      Out8[0][0][9]);
	float res = (Out8[0][0][0]>Out8[0][0][1])?Out8[0][0][0]:Out8[0][0][1];
	
	freeForppptr(Out8,OutSize8);
	// free()
	
	return res;
	
}