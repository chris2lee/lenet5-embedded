
/*
	@author hhy/WatterCutter
	@date 2022/2
	@说明

	输入输出：
	输入输出都为指针，操作根据imageSize类型的尺寸信息参数input/outputSize确定；操作
	的正确性由尺寸参数的正确性保证。因为这是用于在得到神经网络各层输出尺寸以后部署前向
	网络的情况，故而这样的接口设计是合理的。

	动态内存管理：
	只需要对函数内部申请的内存进行释放操作；输入输出的内存申请和释放都应该在函数外部
	由用户完成。除matOperationEdgeExpand()等返回修改后的矩阵的操作，其余操作申请的内
	存应该在函数退出前全部释放。

*/
#include <math.h>
#include "cnnoperation.h"
#include <malloc.h>
#include <stdio.h>

/*
	@func 
		矩阵局部和卷积核的卷积计算
	@params
	@retval 
*/
static float matAndKernelConvolution(float*** mat,imageSize matSize,float*** kernel,    //有点出现问题
	imageSize kernelSize,int nowrow,int nowcol)
{	//输入通道（维度）对应相成相加
	// float res = mat[0][nowrow][nowcol];
	float res = 0.0;
	for (int d = 0;d < kernelSize.dimsc;d++) {//不同维度
		for (int ii = 0;ii < kernelSize.rowsc;ii++) {
		// for (int ii = 0;ii+nowrow < kernelSize.rowsc;ii++) {
			for (int jj = 0;jj < kernelSize.colsc;jj++) {
			// for (int jj = 0;nowcol + jj < kernelSize.colsc;jj++) {
				res += (mat[d][nowrow + ii][nowcol + jj] * kernel[d][ii][jj]);
			}
			// 
		}
	}
	
	return res;
}
/*
	@func: 
		卷积操作
	@params 
		inputmat:输入
		inputSize:输入尺寸
		outputmat:输出
		outputSize:输出尺寸
		kernel:卷积核
		kernelSize:卷积核维度、行数、列数参数；维度应与inputmat的维度一致
		paddding:补边的宽度
		step:卷积步长
*/
void cnnOperationConvolution(float*** inputmat, imageSize inputSize,
	float*** outputmat, imageSize outputSize, float**** kernel, imageSize kernelSize,
	int padding,int step,float* bias)
{
	printf("cnnOperationConvolution start\n");
	
	//加边后的通道
	float*** exMat = (float***)malloc(inputSize.dimsc * sizeof(float**));
	
	imageSize exMatSize;
	exMatSize.rowsc = inputSize.rowsc + 2*padding;
	exMatSize.colsc = inputSize.colsc + 2*padding;
	// exMatSize.rowsc = inputSize.rowsc + padding;
	// exMatSize.colsc = inputSize.colsc + padding;
	exMatSize.dimsc = inputSize.dimsc;

	//matOperationEdgeExpand()将为exMat的不同维度申请内存
	int r = inputSize.rowsc;
	int c = inputSize.colsc;
	
	for (int k = 0;k < inputSize.dimsc;k++) {//不同维度
		//加零边 这里尺寸参数使用inputSize是正确的。两边加padding， exMat = inputmat
		exMat[k] = matOperationEdgeExpand(inputmat[k], inputSize.rowsc,inputSize.colsc, padding, padding);		
	}	
	// printf("exmat is ok!");
	//卷积操作 滑动卷积核
	// for (int n = 0;n < kernelSize.numsc;n++) {// 一个卷积核输出一个通道
	for (int n = 0;n < outputSize.dimsc;n++) {// 一个卷积核输出一个通道
		//移动卷积核；（i,j）是卷积核左上角元素在exMat中的坐标
		for (int i = 0;i < outputSize.rowsc;i+=step) {//不同行
			for (int j = 0;j < outputSize.colsc ;j+=step) {//不同列
				//每个维度上卷积得到多个卷积和，卷积和相加得到输出中的值outputmat[n][i][j]，exMat拓展后
				// outputmat[n][i][j]  =  matAndKernelConvolution(exMat, exMatSize, kernel[n],kernelSize, i, j) ;	
				outputmat[n][i][j]  =  matAndKernelConvolution(exMat, exMatSize, kernel[n],kernelSize, i, j)+bias[n];
			}
		}
	}

	
	// float *vecMulti = (float*)malloc(outputSize.colsc*sizeof(float)); 
	// for (int i = 0;i < outputSize.colsc;i++) {
	// 	vecMulti[i] = vectorMultiplication(inputmat[0][0], weight[i], inputSize.colsc);
	// 	outputmat[0][0][i] = vecMulti[i] + bias[i];
	// }
	// free(vecMulti);

	//此处释放之前为exMat各维度申请的内存
	for (int i = 0;i < kernelSize.dimsc;i++) {
		free(exMat[i]);
	}
	printf("cnnOperationConvolution end\n");
	;
}

/*
	@func 
		最大池化
	@params
		mat:
		matSize:
		kernelSize:
		nowrow:
		nowcol: 
*/
// static float matAndKernelMaxPooling(float*** mat, imageSize matSize,
// 	imageSize kernelSize, int nowrow, int nowcol)
// {	
// 	float max = mat[0][nowrow][nowcol];
// 	for (int d = 0;d < matSize.dimsc;d++) {//不同维度
// 		for (int ii = 0;ii < kernelSize.rowsc;ii++) {  
// 			for (int jj = 0;jj < kernelSize.colsc;jj++) {
// 				max = (max>mat[d][ii][jj])?max:mat[d][ii][jj];
// 			}
// 		}
// 	}				
// 	return max;
// }
static float matAndKernelMaxPooling(float*** mat, imageSize matSize,
	imageSize kernelSize, int nowrow, int nowcol,int d)
{	
	float max = mat[d][2*nowrow][2*nowcol];
	
	for (int ii = 0;ii < kernelSize.rowsc;ii++) {  
		for (int jj = 0;jj < kernelSize.colsc;jj++) {
			max = (max>mat[d][2*nowrow+ii][2*nowcol+jj])?max:mat[d][2*nowrow+ii][2*nowcol+jj];
		}
	}
					
	return max;
}
/*	
	@func:
		池化操作
	@params
		inputmat:输入
		inputSize:输入尺寸
		outputmat:输出
		outputSize:输出尺寸
		kernel:核
		kernelSize:核维度、行数、列数参数
		paddding:补边的宽度
		step:卷积步长
*/
void cnnOperationPooling(float*** inputmat, imageSize inputSize,
	float*** outputmat, imageSize outputSize,
	imageSize kernelSize, int padding, int step)
{
	printf("cnnOperationPooling start\n");	
	//加边后的通道
	float*** exMat = (float***)malloc(inputSize.dimsc * sizeof(float**));
	imageSize exMatSize;
	exMatSize.rowsc = inputSize.rowsc + 2*padding; //宽加padding
	exMatSize.colsc = inputSize.colsc + 2*padding; //长加padding
	exMatSize.dimsc = inputSize.dimsc;           //通道数
	
	int r = inputSize.rowsc;    //输入行数
	int c = inputSize.colsc;    //输入列数
	//matOperationEdgeExpand()将为exMat的不同维度申请内存
	for (int k = 0;k < inputSize.dimsc;k++) {//不同维度
		//加零边 这里尺寸参数使用inputSize是正确的
		exMat[k] = matOperationEdgeExpand(inputmat[k], inputSize.rowsc,inputSize.colsc, padding, padding);
	}
	//池化操作 滑动核
	// for (int n = 0;n < kernelSize.numsc;n++) {//不同核	
	for (int n = 0;n < outputSize.dimsc;n++) {//不同核	
		// int rlimit = exMatSize.rowsc-step-kernelSize.rowsc-1;
		// int climit = exMatSize.rowsc-step-kernelSize.colsc-1;
		// 没啥用  没啥用………………
		// int rlimit = exMatSize.rowsc-kernelSize.rowsc;
		// int climit = exMatSize.colsc-kernelSize.colsc;
		//
		//移动核；（i,j）是核左上角元素在exMat中的坐标
		for (int i = 0;i < outputSize.rowsc;i+=step) {//不同行
			for (int j = 0;j < outputSize.colsc ;j+=step) {//不同列
				outputmat[n][i][j] = matAndKernelMaxPooling(exMat, exMatSize, kernelSize, i, j,n);	
			}
		}
	}
	//此处释放之前为exMat各维度申请的内存
	for (int i = 0;i < kernelSize.dimsc;i++) {
		free(exMat[i]);
	}
	printf("cnnOperationPooling end\n");	
}

/* 
	@func
		扁平化操作
	@params 
		input:
		inputsize:
		output:
		outputsize:
*/
//特征图展开成一维向量
void cnnOperationFlatten(float*** input, imageSize inputSize, float*** output,
	imageSize outputSize)
{
	printf("cnnOperationFlatten start\n");
	int s = inputSize.rowsc * inputSize.colsc;      //通道具有的元素个数
	int length = inputSize.rowsc * inputSize.colsc * inputSize.dimsc; //总共有length个元素
	for(int n=0;n<inputSize.dimsc;n++)
		for(int i=0;i<inputSize.rowsc;i++)
			for (int j = 0;j < inputSize.colsc;j++) {
				// int ptr = (n * s + i * inputSize.rowsc + j);
				int ptr = (n * s + i * inputSize.colsc + j); 
				// output[0][0][ptr] = input[n][i][j]
				output[0][0][ptr] = input[n][i][j];
			}
	printf("cnnOperationFlatten end\n");
}

/*
	@func
		relu激活函数
	@params
		 
*/
void cnnOperationActivation(float*** inputmat, imageSize inputSize,float bias)
{
	printf("cnnOperationActivation start\n");
	for (int n = 0;n < inputSize.dimsc;n++)
		for (int i = 0;i < inputSize.rowsc;i++)
			for (int j = 0;j < inputSize.colsc;j++) {
				inputmat[n][i][j] = (inputmat[n][i][j]>0)?inputmat[n][i][j]:0;
			}







	printf("cnnOperationActivation end\n");
}
// void ReLUActivation(float*** inputmat, imageSize inputSize)
// {
// 	printf("ReLU start\n");
// 	for (int n = 0;n < inputSize.dimsc;n++)
// 		for (int i = 0;i < inputSize.rowsc;i++)
// 			for (int j = 0;j < inputSize.colsc;j++) {
// 				inputmat[n][i][j] = (inputmat[n][i][j]>0)?inputmat[n][i][j]:0;
// 			}

// 	printf("ReLU end\n");
// }
// void SigmoidActivation(float*** inputmat, imageSize inputSize)
// {
// 	printf("Sigmoid start\n");
// 	for (int n = 0;n < inputSize.dimsc;n++)
// 		for (int i = 0;i < inputSize.rowsc;i++)
// 			for (int j = 0;j < inputSize.colsc;j++) {
// 				inputmat[n][i][j] = 1/(1+exp(-inputmat[n][i][j]));
// 			}

// 	printf("Sigmoid end\n");
// }
/*
	@func
		向量叉乘
	@params
		 
*/ 
static float vectorMultiplication(float* vec1,float* vec2,int len)
{
	float res = 0;
	for (int i = 0;i < len;i++) {
		res += vec1[i] * vec2[i];
	}
	return res;
}
/*
	@func
		全连接处理
	@params
*/
void cnnOperationLinear(float*** inputmat, imageSize inputSize,
	float*** outputmat, imageSize outputSize,float** weight,float* bias)
{
	printf("cnnOperationLinear start\n");
	float *vecMulti = (float*)malloc(outputSize.colsc*sizeof(float)); 
	for (int i = 0;i < outputSize.colsc;i++) {
		vecMulti[i] = vectorMultiplication(inputmat[0][0], weight[i], inputSize.colsc);
		outputmat[0][0][i] = vecMulti[i] + bias[i];
	}
	free(vecMulti);
	printf("cnnOperationLinear end\n");
}