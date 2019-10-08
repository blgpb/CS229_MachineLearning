/*
Copyright (c) 2019 All rights reserved.
Created on    2019-10-7 21:46:44
Author:       Wentao Yu
Version:      1.0
Describition: 线性回归，使用梯度下降法（只针对一个输入训练样本）
*/

#include <stdio.h>

void Gradient_Descent(double* x_input, double y_output, int x_input_dimension, double** theta_learned, double learning_rate, double converge)
{
	int i;
	double* theta_learned_temp = NULL;
	double h_theta = 0.0;
	double J_theta[2] = { 0.0, 0.0 };//目标函数值，第一个位置存放上一次迭代后的目标函数值，第二个位置存放这一次迭代后的目标函数值
	int iteration_count = 0;

	theta_learned_temp = (double*)malloc(sizeof(double) * x_input_dimension);
	//系数一开始全部初始化为0
	for (i = 0; i < x_input_dimension; i++)
	{
		theta_learned_temp[i] = 0.0;
	}
	//计算初始的h_theta
	for (i = 0; i < x_input_dimension; i++)
	{
		h_theta = h_theta + x_input[i] * theta_learned_temp[i];
	}
	//计算初始的J_theta_last
	J_theta[iteration_count % 2] = 0.5 * (h_theta - y_output) * (h_theta - y_output);

	while (((J_theta[iteration_count % 2] - J_theta[(iteration_count + 1) % 2]) * (J_theta[iteration_count % 2] - J_theta[(iteration_count + 1) % 2])) > (converge * converge))
	{
		//梯度下降法更新公式
		for (i = 0; i < x_input_dimension; i++)
		{
			theta_learned_temp[i] = theta_learned_temp[i] - learning_rate * (h_theta - y_output) * x_input[i];
		}

		//计算h_theta
		h_theta = 0.0;
		for (i = 0; i < x_input_dimension; i++)
		{
			h_theta = h_theta + x_input[i] * theta_learned_temp[i];
		}
		J_theta[(iteration_count + 1) % 2] = 0.5 * (h_theta - y_output) * (h_theta - y_output);

		//迭代次数加1
		iteration_count++;
		printf("第%d次迭代的h_theta:%f,目标函数值:%f\n", iteration_count, h_theta, J_theta[1]);
	}

	*theta_learned = theta_learned_temp;

}

void free_mem(double** p)
{
	if (*p != NULL)
	{
		free(*p);
		*p = NULL;
	}
}
void main()
{
	//假设一个线性回归符合 y=0.1 * x0 + 0.2 * x1 + 0.3 * x2 + 0.4 * x3 + 0.5*x4 + 0.6*x5
	//上式中的系数0.1 0.2 0.3 0.4 0.5 0.6是隐藏的pattern，需要本程序线性回归求得

	double x_input[] = { 1,2,3,4,5,6 };
	double theta[] = { 0.1,0.2,0.3,0.4,0.5,0.6 };//这是隐藏的pattern，需要本程序线性回归求得
	double y_output = 0.0;
	int i;
	int x_input_dimension = 0;
	
	//与机器学习算法相关的参数
	double learning_rate = 0.01;//梯度下降法的学习率
	double converge = 0.01;//目标函数收敛的参数(迭代前后目标函数的差值)
	double* theta_learned = NULL;

	x_input_dimension = (int)(sizeof(x_input) / sizeof(double));

	for (i = 0; i < x_input_dimension; i++)
	{
		y_output = y_output + x_input[i] * theta[i];
	}
	printf("给定样本中y=%f\n", y_output);

	//调用梯度下降算法
	Gradient_Descent(x_input, y_output, x_input_dimension, &theta_learned, learning_rate, converge);
	
	for (i = 0; i < x_input_dimension; i++)
	{
		printf("线性回归后得到的theta[%d]'=%f\n", i, theta_learned[i]);
	}

	y_output = 0;
	for (i = 0; i < x_input_dimension; i++)
	{
		y_output = y_output + x_input[i] * theta_learned[i];
	}

	printf("线性回归后得到的y'=%f",y_output);
	
	//free memory
	free_mem(&theta_learned);
	
}