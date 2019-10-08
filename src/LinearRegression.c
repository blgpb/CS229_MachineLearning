/*
Copyright (c) 2019 All rights reserved.
Created on    2019-10-7 21:46:44
Author:       Wentao Yu
Version:      1.0
Describition: ���Իع飬ʹ���ݶ��½�����ֻ���һ������ѵ��������
*/

#include <stdio.h>

void Gradient_Descent(double* x_input, double y_output, int x_input_dimension, double** theta_learned, double learning_rate, double converge)
{
	int i;
	double* theta_learned_temp = NULL;
	double h_theta = 0.0;
	double J_theta[2] = { 0.0, 0.0 };//Ŀ�꺯��ֵ����һ��λ�ô����һ�ε������Ŀ�꺯��ֵ���ڶ���λ�ô����һ�ε������Ŀ�꺯��ֵ
	int iteration_count = 0;

	theta_learned_temp = (double*)malloc(sizeof(double) * x_input_dimension);
	//ϵ��һ��ʼȫ����ʼ��Ϊ0
	for (i = 0; i < x_input_dimension; i++)
	{
		theta_learned_temp[i] = 0.0;
	}
	//�����ʼ��h_theta
	for (i = 0; i < x_input_dimension; i++)
	{
		h_theta = h_theta + x_input[i] * theta_learned_temp[i];
	}
	//�����ʼ��J_theta_last
	J_theta[iteration_count % 2] = 0.5 * (h_theta - y_output) * (h_theta - y_output);

	while (((J_theta[iteration_count % 2] - J_theta[(iteration_count + 1) % 2]) * (J_theta[iteration_count % 2] - J_theta[(iteration_count + 1) % 2])) > (converge * converge))
	{
		//�ݶ��½������¹�ʽ
		for (i = 0; i < x_input_dimension; i++)
		{
			theta_learned_temp[i] = theta_learned_temp[i] - learning_rate * (h_theta - y_output) * x_input[i];
		}

		//����h_theta
		h_theta = 0.0;
		for (i = 0; i < x_input_dimension; i++)
		{
			h_theta = h_theta + x_input[i] * theta_learned_temp[i];
		}
		J_theta[(iteration_count + 1) % 2] = 0.5 * (h_theta - y_output) * (h_theta - y_output);

		//����������1
		iteration_count++;
		printf("��%d�ε�����h_theta:%f,Ŀ�꺯��ֵ:%f\n", iteration_count, h_theta, J_theta[1]);
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
	//����һ�����Իع���� y=0.1 * x0 + 0.2 * x1 + 0.3 * x2 + 0.4 * x3 + 0.5*x4 + 0.6*x5
	//��ʽ�е�ϵ��0.1 0.2 0.3 0.4 0.5 0.6�����ص�pattern����Ҫ���������Իع����

	double x_input[] = { 1,2,3,4,5,6 };
	double theta[] = { 0.1,0.2,0.3,0.4,0.5,0.6 };//�������ص�pattern����Ҫ���������Իع����
	double y_output = 0.0;
	int i;
	int x_input_dimension = 0;
	
	//�����ѧϰ�㷨��صĲ���
	double learning_rate = 0.01;//�ݶ��½�����ѧϰ��
	double converge = 0.01;//Ŀ�꺯�������Ĳ���(����ǰ��Ŀ�꺯���Ĳ�ֵ)
	double* theta_learned = NULL;

	x_input_dimension = (int)(sizeof(x_input) / sizeof(double));

	for (i = 0; i < x_input_dimension; i++)
	{
		y_output = y_output + x_input[i] * theta[i];
	}
	printf("����������y=%f\n", y_output);

	//�����ݶ��½��㷨
	Gradient_Descent(x_input, y_output, x_input_dimension, &theta_learned, learning_rate, converge);
	
	for (i = 0; i < x_input_dimension; i++)
	{
		printf("���Իع��õ���theta[%d]'=%f\n", i, theta_learned[i]);
	}

	y_output = 0;
	for (i = 0; i < x_input_dimension; i++)
	{
		y_output = y_output + x_input[i] * theta_learned[i];
	}

	printf("���Իع��õ���y'=%f",y_output);
	
	//free memory
	free_mem(&theta_learned);
	
}