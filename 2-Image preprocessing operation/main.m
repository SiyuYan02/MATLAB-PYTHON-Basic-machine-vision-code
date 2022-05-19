
imag = imread('npy.jpg');  %读取图片
imag = rgb2gray(imag);        %转化为灰度图
figure(1),subplot(151),imshow(imag),title('原图'); 
[high,width] = size(imag);   % 获得图像的高度和宽度

%%sobel边缘检测
F2 = double(imag);        
U = double(imag); 
uSobel = imag;
T=250;
for i = 2:high - 1   %sobel边缘检测
    for j = 2:width - 1
        Gx = (U(i+1,j-1) + 2*U(i+1,j) + F2(i+1,j+1)) - (U(i-1,j-1) + 2*U(i-1,j) + F2(i-1,j+1));
        Gy = (U(i-1,j+1) + 2*U(i,j+1) + F2(i+1,j+1)) - (U(i-1,j-1) + 2*U(i,j-1) + F2(i+1,j-1));
        uSobel(i,j) = sqrt(Gx^2 + Gy^2); 
        if uSobel(i,j)<T
            uSobel(i,j)=0;
        else
            uSobel(i,j)=255;
        end
    end
end 
subplot(152);imshow(im2uint8(uSobel)),title('sobel算子边缘检测图像');  


%%拉普拉斯边缘检测
w=fspecial('gaussian',[5 5]);   
I=imfilter(imag,w,'replicate');%平滑
[m,n]=size(I);
Ig=I;
for i=2:m-1
    for j=2:n-1
        Ig(i,j)=(1+4).*I(i,j)-(I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1));
        %Ig(i,j)=sum(sum(Ig));
    end
end
subplot(153),imshow(uint8(Ig)),title('拉普拉斯边缘检测图像');
Ig_=Ig+I;
subplot(154),imshow(uint8(Ig_)),title('拉普拉斯锐化后的图像');

%高斯滤波后使用Canny算子进行边缘检测
h=fspecial('gaussian',5);%高斯滤波
I2=imfilter(imag,h,'replicate');
GcannyBW=edge(I2,'canny');
subplot(155);imshow(im2uint8(GcannyBW)),title('canny算子边缘检测');  

%产生运动模糊
len=20;%设置运动位移为35个像素
theta=25;%设置运动角度为45度
psf=fspecial('motion',len,theta);%建立二维运动仿真滤波器psf
mf=imfilter(imag,psf,'circular','conv');%用psf产生退化图像
figure(2)
subplot(141),imshow(mf),title('加入运动模糊的图像');

%图像增强
[m,n]=size(mf);
mf_g=mf;
for i=2:m-1
    for j=2:n-1
        mf_g(i,j)=(1+4).*mf(i,j)-(mf(i+1,j)+mf(i-1,j)+mf(i,j+1)+mf(i,j-1));
        %Ig(i,j)=sum(sum(Ig));
    end
end
mf_g=mf_g+mf;
subplot(142),imshow(uint8(mf_g)),title('拉普拉算子斯实现图像增强');

%维纳滤波实现图像复原
mnr1=deconvwnr(mf,psf,0);
subplot(143),imshow(uint8(mnr1)),title('维纳滤波实现图像复原');
filt = 1/25 * ones(5);
C = imfilter(mnr1,filt,'symmetric','same');
subplot(144),imshow(uint8(C)),title('均值滤波去噪');

%改变对比度与亮度
figure(3);
imag_1=adjust(imag,0.5,0);
subplot(231),imshow(uint8(imag_1)),title('对比度为0.5');

imag_2=adjust(imag,1,0);
subplot(232),imshow(uint8(imag_2)),title('对比度为1');

imag_3=adjust(imag,2,0);
subplot(233),imshow(uint8(imag_3)),title('对比度为2');

%改变亮度
imag_4=adjust(imag,0.6,20);
subplot(234),imshow(uint8(imag_4)),title('亮度为20');

imag_5=adjust(imag,0.6,60);
subplot(235),imshow(uint8(imag_5)),title('亮度为60');

imag_6=adjust(imag,0.6,100);
subplot(236),imshow(uint8(imag_6)),title('亮度为100');




