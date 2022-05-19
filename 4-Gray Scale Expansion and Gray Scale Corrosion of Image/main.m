
I=im2double(imread('small-blobs.tif'));
figure(1),subplot(1,3,1),imshow(I),title('原图'); 
B=[1 1 1;1 1 1;1 1 1]; 
J_Erosion=corrode(I,B);
J_Dilation=expand(I,B);
subplot(1,3,2),imshow(J_Erosion),title('腐蚀后图像');   
subplot(1,3,3),imshow(J_Dilation),title('膨胀后图像');  

I_2=imread('small-blobs-gradient.tif');
figure(2),subplot(1,2,1),imshow(I_2),title('梯度幅值图像'); 
%开运算
I_3=imopen(I_2,B);
%闭运算
I_3=imclose(I_3,B);
subplot(1,2,2),imshow(I_3),title('形态学平滑')

%对原始梯度幅值图像进行分水岭分割
I_4=watershed(I_2);
I_4=label2rgb(I_4);%并将生成的标签矩阵显示为RGB图像
I_4=rgb2gray(I_4);        
figure(3);
subplot(1,2,1),imshow(I_4),title('原始梯度幅值图像的分水岭分割结果')

%对平滑后的梯度幅值图进行分水岭分割
I_5=watershed(I_3);
I_5=label2rgb(I_5);%并将生成的标签矩阵显示为RGB图像
I_5=rgb2gray(I_5);      
subplot(1,2,2),imshow(I_5),title('平滑后的梯度幅值图的分水岭分割结果')
