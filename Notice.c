/*
	@author hhy/WattercCutter
	@func to mark some notice of programming
*/

/* start 动态分配内存的高维矩阵空间可以用指针传参 */ 
void ppptrInitWithArray(float**** ptr){
	ptr[0] = (float***)malloc(sizeof(float**));
	ptr[0][0] = (float**)malloc(sizeof(float*));
	ptr[0][0][0] = (float*)malloc(sizeof(float));
	ptr[0][0][0][0] = 88;
}
int main()
{
	/* 
	初始化时必须分配内存空间 
		float**** ptr = NULL;这种不能作为参数传递。
	*/
	float**** ptr = (float****)malloc(sizeof(float***));;	
	ppptrInitWithArray(ptr);

	printf("%f",ptr[0][0][0][0]);
}

/* end 动态分配内存的高维矩阵空间可以用指针传参 */ 