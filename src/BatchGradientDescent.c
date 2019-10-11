/*
Copyright (c) 2019 All rights reserved.
Created on    2019-10-8 20:04:07
Author:       Wentao Yu
Version:      1.0
Describition: 线性回归，使用批梯度下降法（针对m个输入训练样本）
*/

#include <stdio.h>
#include <stdlib.h>

void create_train_examples(double*** x_input, double** y_output, double* theta, int examples_num, int x_input_dimension)
{
	double** x_input_temp = NULL;
	double* y_output_temp = NULL;
	int i,j;

	x_input_temp = (double**)malloc(sizeof(double*) * examples_num);
	y_output_temp = (double*)malloc(sizeof(double) * examples_num);

	//将训练样本中的每一个y初始化
	for (i= 0; i < examples_num; i++)
	{
		y_output_temp[i] = 0;
	}

	for (i = 0; i < examples_num; i++)
	{
		x_input_temp[i] = (double*)malloc(sizeof(double) * x_input_dimension);
		for (j = 0; j < x_input_dimension; j++)
		{
			x_input_temp[i][j] = (rand() % 6);//每一个训练样本中的x的数字都是随机初始化的
			y_output_temp[i] = x_input_temp[i][j] * theta[j] + y_output_temp[i];
		}
	}

	*x_input = x_input_temp;
	*y_output = y_output_temp;
}

void printf_created_train_examples(double** x_input, double* y_output, int examples_num, int x_input_dimension)
{
	int i,j;

	printf("创建的训练样本\n");
	for (i = 0; i < examples_num; i++)
	{
		printf("第%d个训练样本:\n", i);
		printf("x的值为:\n");
		for (j = 0; j < x_input_dimension; j++)
		{
			printf("%f ", x_input[i][j]);
		}
		printf("y的值为:%f\n", y_output[i]);
	}
}
void Batch_Gradient_Descent(double** x_input, double* y_output, int x_input_dimension, double** theta_learned, int examples_num, double learning_rate, double converge)
{
	int i,j;
	double* theta_learned_temp = NULL;
	double* h_theta = NULL;
	double J_theta[2] = { 0.0, 0.0 };//目标函数值，第一个位置存放上一次迭代后的目标函数值，第二个位置存放这一次迭代后的目标函数值
	int iteration_count = 0;

	theta_learned_temp = (double*)malloc(sizeof(double) * x_input_dimension);

	h_theta = (double*)malloc(sizeof(double) * examples_num);
	//系数一开始全部初始化为0
	for (i = 0; i < x_input_dimension; i++)
	{
		theta_learned_temp[i] = 0.0;
	}
	//h_theta初始化为0
	for (i = 0; i < examples_num; i++)
	{
		h_theta[i] = 0.0;
	}
	//计算初始的h_theta
	for (i = 0; i < examples_num; i++)
	{
		for (j = 0; j < x_input_dimension; j++)
		{
			h_theta[i] = h_theta[i] + x_input[i][j] * theta_learned_temp[i];
		}
	}
	//计算初始的J_theta_last
	for (i = 0; i < examples_num; i++)
	{
		J_theta[iteration_count % 2] = 0.5 * (h_theta[i] - y_output[i]) * (h_theta[i] - y_output[i]) + J_theta[iteration_count % 2];
	}

	while (((J_theta[iteration_count % 2] - J_theta[(iteration_count + 1) % 2]) * (J_theta[iteration_count % 2] - J_theta[(iteration_count + 1) % 2])) > (converge * converge))
	{
		//梯度下降法更新公式
		for (i = 0; i < x_input_dimension; i++)
		{
			for (j = 0; j < examples_num; j++)
			{
				theta_learned_temp[i] = theta_learned_temp[i] - learning_rate * (h_theta[j] - y_output[j]) * x_input[j][i];
			}
		}

		//h_theta清0
		for (i = 0; i < examples_num; i++)
		{
			h_theta[i] = 0.0;
		}
		//计算每个训练样本预测得到的h_theta
		for (i = 0; i < examples_num; i++)
		{
			for (j = 0; j < x_input_dimension; j++)
			{
				h_theta[i] = h_theta[i] + x_input[i][j] * theta_learned_temp[i];
			}
		}

		//J_theta清0
		J_theta[(iteration_count + 1) % 2] = 0.0;

		//计算迭代后得到的J_theta
		for (i = 0; i < examples_num; i++)
		{
			J_theta[(iteration_count + 1) % 2] = 0.5 * (h_theta[i] - y_output[i]) * (h_theta[i] - y_output[i]) + J_theta[(iteration_count + 1) % 2];
		}

		//迭代次数加1
		iteration_count++;
		printf("第%d次迭代的第1个样本预测得到的h_theta:%f,目标函数值:%f\n", iteration_count, h_theta[1], J_theta[1]);
	}

	*theta_learned = theta_learned_temp;

}

void free_mem_2_pointer(double*** p, int examples_num)
{
	int i;

	for (i = 0; i < examples_num; i++)
	{
		if ((*p)[i] != NULL)
		{
			free((*p)[i]);
			(*p)[i] = NULL;
		}
	}
	if (*p != NULL)
	{
		free(*p);
		*p = NULL;
	}
}

void free_mem_1_pointer(double** p)
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

	//由于具有m个输入训练样本，需要二维数组，本程序采用手工打造二维数组，用二级指针做输入
	double** x_input = NULL;
	double* y_output = NULL;
	double theta[] = { 0.1,0.2,0.3,0.4,0.5,0.6 };//这是隐藏的pattern，需要本程序线性回归求得
	int x_input_dimension = 6;
	int examples_num = 10;
	int i;

	//与机器学习算法相关的参数
	double learning_rate = 0.000001;//梯度下降法的学习率
	double converge = 0.01;//目标函数收敛的参数(迭代前后目标函数的差值)
	double* theta_learned = NULL;

	//创建训练数据
	create_train_examples(&x_input, &y_output, theta, examples_num, x_input_dimension);
	//打印创建的训练数据
	printf_created_train_examples(x_input, y_output, examples_num, x_input_dimension);
	
	//调用梯度下降算法
	Batch_Gradient_Descent(x_input, y_output, x_input_dimension, &theta_learned, examples_num, learning_rate, converge);
	
	for (i = 0; i < x_input_dimension; i++)
	{
		printf("线性回归后得到的theta[%d]'=%f\n", i, theta_learned[i]);
	}

	//free memory
	free_mem_2_pointer(&x_input, examples_num);
	free_mem_1_pointer(&y_output);
	free_mem_1_pointer(&theta_learned);

}