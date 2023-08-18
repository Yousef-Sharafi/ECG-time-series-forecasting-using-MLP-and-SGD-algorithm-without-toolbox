%Programmer: Yousef Sharafi

clc;
close all;
clear all;

data=xlsread('ECG.xlsx');
num_data=size(data,1);

percent_train=0.75;
num_train=round(num_data*percent_train);
num_test=num_data-num_train;

n1=3;
n2=5;
n3=3;
n4=1;

eta=0.05;

epoch=20;
mse_train=zeros(1,epoch);
mse_test=zeros(1,epoch);

a=-1;
b=1;

w1=unifrnd(a,b,[n2 n1]);
net1=zeros(n2,1);
o1=zeros(n2,1);

w2=unifrnd(a,b,[n3 n2]);
net2=zeros(n3,1);
o2=zeros(n3,1);

w3=unifrnd(a,b,[n4 n3]);
net3=zeros(n4,1);
o3=zeros(n4,1);

for t=1:epoch  
        error=zeros(1,num_train);
        for i=1:num_train
            input=data(i,1:3);
            net1=w1*input';
            o1=logsig(net1);
            net2=w2*o1;
            o2=logsig(net2);
            net3=w3*o2;
            o3=(net3);
            target=data(i,4);
            error(i)=target-o3;

            t1=o2.*(1-o2);
            A=diag(t1);

            t2=o1.*(1-o1);
            B=diag(t2);

            w1=w1-eta*error(i)*-1*(w3*A*w2*B)'*input;
            w2=w2-eta*error(i)*-1*(w3*A)'*o1';
            w3=w3-eta*error(i)*-1*1*o2';
        end

        error_data_train=zeros(1,num_train);
        output_data_train=zeros(1,num_train);
        for i=1:num_train
            input=data(i,1:3);
            net1=w1*input';
            o1=logsig(net1);
            net2=w2*o1;
            o2=logsig(net2);
            net3=w3*o2;
            o3=(net3);
            target=data(i,4);
            output_data_train(i)=o3;
            error_data_train(i)=target-o3;
        end
        mse_train(t)=mse(error_data_train);

        error_data_test=zeros(1,num_test);
        output_data_test=zeros(1,num_test);
        for i=1:num_test
            input=data(num_train+i,1:3);
            net1=w1*input';
            o1=logsig(net1);
            net2=w2*o1;
            o2=logsig(net2);
            net3=w3*o2;
            o3=(net3);
            target=data(num_train+i,4);
            output_data_test(i)=o3;
            error_data_test(i)=target-o3;
        end
        mse_test(t)=mse(error_data_test);

        figure(1);
        subplot(2,2,1),plot(data(1:num_train,4));
        hold on;
        plot(output_data_train,'r','linewidth',1);
        hold off;
        xlabel('Train Data');
        ylabel('Output');

        subplot(2,2,2),semilogy(mse_train(1:t));
        hold off;
        xlabel('Epoch');
        ylabel('mse train');

        subplot(2,2,3),plot(data(num_train+1:num_data,4));
        hold on;
        plot(output_data_test,'r','linewidth',1);
        hold off;
        xlabel('Test Data');
        ylabel('Output');

        subplot(2,2,4),semilogy(mse_test(1:t));
        hold off;
        xlabel('Epoch');
        ylabel('mse test');   
end

figure(2);
plotregression(data(1:num_train,4),output_data_train);
title('Regression Train');

figure(3);
plotregression(data(num_train+1:num_data,4),output_data_test);
title('Regression Test');

mse_train_result = mse_train(epoch)
mse_test_result = mse_test(epoch)




