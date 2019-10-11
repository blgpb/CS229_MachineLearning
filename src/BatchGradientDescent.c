/*
Copyright (c) 2019 All rights reserved.
Created on    2019-10-8 20:04:07
Author:       Wentao Yu
Version:      1.0
Describition: ���Իع飬ʹ�����ݶ��½��������m������ѵ��������
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

	//��ѵ�������е�ÿһ��y��ʼ��
	for (i= 0; i < examples_num; i++)
	{
		y_output_temp[i] = 0;
	}

	for (i = 0; i < examples_num; i++)
	{
		x_input_temp[i] = (double*)malloc(sizeof(double) * x_input_dimension);
		for (j = 0; j < x_input_dimension; j++)
		{
			x_input_temp[i][j] = (rand() % 6);//ÿһ��ѵ�������е�x�����ֶ��������ʼ����
			y_output_temp[i] = x_input_temp[i][j] * theta[j] + y_output_temp[i];
		}
	}

	*x_input = x_input_temp;
	*y_output = y_output_temp;
}

void printf_created_train_examples(double** x_input, double* y_output, int examples_num, int x_input_dimension)
{
	int i,j;

	printf("������ѵ������\n");
	for (i = 0; i < examples_num; i++)
	{
		printf("��%d��ѵ������:\n", i);
		printf("x��ֵΪ:\n");
		for (j = 0; j < x_input_dimension; j++)
		{
			printf("%f ", x_input[i][j]);
		}
		printf("y��ֵΪ:%f\n", y_output[i]);
	}
}
void Batch_Gradient_Descent(double** x_input, double* y_output, int x_input_dimension, double** theta_learned, int examples_num, double learning_rate, double converge)
{
	int i,j;
	double* theta_learned_temp = NULL;
	double* h_theta = NULL;
	double J_theta[2] = { 0.0, 0.0 };//Ŀ�꺯��ֵ����һ��λ�ô����һ�ε������Ŀ�꺯��ֵ���ڶ���λ�ô����һ�ε������Ŀ�꺯��ֵ
	int iteration_count = 0;

	theta_learned_temp = (double*)malloc(sizeof(double) * x_input_dimension);

	h_theta = (double*)malloc(sizeof(double) * examples_num);
	//ϵ��һ��ʼȫ����ʼ��Ϊ0
	for (i = 0; i < x_input_dimension; i++)
	{
		theta_learned_temp[i] = 0.0;
	}
	//h_theta��ʼ��Ϊ0
	for (i = 0; i < examples_num; i++)
	{
		h_theta[i] = 0.0;
	}
	//�����ʼ��h_theta
	for (i = 0; i < examples_num; i++)
	{
		for (j = 0; j < x_input_dimension; j++)
		{
			h_theta[i] = h_theta[i] + x_input[i][j] * theta_learned_temp[i];
		}
	}
	//�����ʼ��J_theta_last
	for (i = 0; i < examples_num; i++)
	{
		J_theta[iteration_count % 2] = 0.5 * (h_theta[i] - y_output[i]) * (h_theta[i] - y_output[i]) + J_theta[iteration_count % 2];
	}

	while (((J_theta[iteration_count % 2] - J_theta[(iteration_count + 1) % 2]) * (J_theta[iteration_count % 2] - J_theta[(iteration_count + 1) % 2])) > (converge * converge))
	{
		//�ݶ��½������¹�ʽ
		for (i = 0; i < x_input_dimension; i++)
		{
			for (j = 0; j < examples_num; j++)
			{
				theta_learned_temp[i] = theta_learned_temp[i] - learning_rate * (h_theta[j] - y_output[j]) * x_input[j][i];
			}
		}

		//h_theta��0
		for (i = 0; i < examples_num; i++)
		{
			h_theta[i] = 0.0;
		}
		//����ÿ��ѵ������Ԥ��õ���h_theta
		for (i = 0; i < examples_num; i++)
		{
			for (j = 0; j < x_input_dimension; j++)
			{
				h_theta[i] = h_theta[i] + x_input[i][j] * theta_learned_temp[i];
			}
		}

		//J_theta��0
		J_theta[(iteration_count + 1) % 2] = 0.0;

		//���������õ���J_theta
		for (i = 0; i < examples_num; i++)
		{
			J_theta[(iteration_count + 1) % 2] = 0.5 * (h_theta[i] - y_output[i]) * (h_theta[i] - y_output[i]) + J_theta[(iteration_count + 1) % 2];
		}

		//����������1
		iteration_count++;
		printf("��%d�ε����ĵ�1������Ԥ��õ���h_theta:%f,Ŀ�꺯��ֵ:%f\n", iteration_count, h_theta[1], J_theta[1]);
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
	//����һ�����Իع���� y=0.1 * x0 + 0.2 * x1 + 0.3 * x2 + 0.4 * x3 + 0.5*x4 + 0.6*x5
	//��ʽ�е�ϵ��0.1 0.2 0.3 0.4 0.5 0.6�����ص�pattern����Ҫ���������Իع����

	//���ھ���m������ѵ����������Ҫ��ά���飬����������ֹ������ά���飬�ö���ָ��������
	double** x_input = NULL;
	double* y_output = NULL;
	double theta[] = { 0.1,0.2,0.3,0.4,0.5,0.6 };//�������ص�pattern����Ҫ���������Իع����
	int x_input_dimension = 6;
	int examples_num = 10;
	int i;

	//�����ѧϰ�㷨��صĲ���
	double learning_rate = 0.000001;//�ݶ��½�����ѧϰ��
	double converge = 0.01;//Ŀ�꺯�������Ĳ���(����ǰ��Ŀ�꺯���Ĳ�ֵ)
	double* theta_learned = NULL;

	//����ѵ������
	create_train_examples(&x_input, &y_output, theta, examples_num, x_input_dimension);
	//��ӡ������ѵ������
	printf_created_train_examples(x_input, y_output, examples_num, x_input_dimension);
	
	//�����ݶ��½��㷨
	Batch_Gradient_Descent(x_input, y_output, x_input_dimension, &theta_learned, examples_num, learning_rate, converge);
	
	for (i = 0; i < x_input_dimension; i++)
	{
		printf("���Իع��õ���theta[%d]'=%f\n", i, theta_learned[i]);
	}

	//free memory
	free_mem_2_pointer(&x_input, examples_num);
	free_mem_1_pointer(&y_output);
	free_mem_1_pointer(&theta_learned);

}