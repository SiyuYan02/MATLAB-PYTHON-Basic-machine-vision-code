
%图像读取
img=imread("people.bmp");
img_s=imread("scenery.png");

%图像的显示
figure(1)
subplot(1,2,1),imshow(img),title('人像图片');
subplot(1,2,2),imshow(img_s),title('风景图片');
 
%图像的储存，以人像图为例
imwrite(img,"meinv.jpg")

%%将图像灰度化
img_1=rgb2gray(img);
figure(2),subplot(1,2,1),imshow(img_1),title('灰度图像')

%%将图像灰度化
img_2=rgb2gray(img_s);
figure(2),subplot(1,2,2),imshow(img_2),title('灰度图像')

%设置高斯模板并卷积(人像图片）
figure(4);
[rows,cols]=size(img_1);
for i=1:5
    sigma=0.5*i;
    gausFilter=fspecial('gaussian',[rows cols],sigma);
    img_3=imfilter(img,gausFilter,'conv');%滤波后的图像
    subplot(1,5,i),imshow(img_3);
end

%设置高斯模板并卷积(风景图片）
figure(5);
[rows_,cols_]=size(img_2);
for i=1:5
    sigma=2*i;
    gausFilter=fspecial('gaussian',[rows_ cols_],sigma);
    img_4=imfilter(img_2,gausFilter,'conv');%滤波后的图像
    subplot(1,5,i),imshow(img_4);
end
    
%高斯金字塔(人像）
[m,n]=size(img_1);
w=fspecial('gaussian',[3 3]);
img2=imresize(imfilter(img_1,w),[m/2 n/2]);
img3=imresize(imfilter(img2,w),[m/4 n/4]);
img4=imresize(imfilter(img3,w),[m/8 n/8]);
img5=imresize(imfilter(img4,w),[m/16 n/16]);
figure(6);
subplot(1,5,1),imshow(img_1),title('高斯金字塔');
subplot(1,5,2),imshow(img2);
subplot(1,5,3),imshow(img3);
subplot(1,5,4),imshow(img4);
subplot(1,5,5),imshow(img5);

%高斯金字塔(风景）
[m,n]=size(img_2);
w=fspecial('gaussian',[3 3]);
img2=imresize(imfilter(img_2,w),[m/2 n/2]);
img3=imresize(imfilter(img2,w),[m/4 n/4]);
img4=imresize(imfilter(img3,w),[m/8 n/8]);
img5=imresize(imfilter(img4,w),[m/16 n/16]);
figure(7);
subplot(1,5,1),imshow(img_2),title('高斯金字塔');
subplot(1,5,2),imshow(img2);
subplot(1,5,3),imshow(img3);
subplot(1,5,4),imshow(img4);
subplot(1,5,5),imshow(img5);

% 加高斯噪声
Image_noise_gauss = imnoise(img_1,'gaussian'); %加噪
Image_noise_gauss_ = imnoise(img_2,'gaussian'); %加噪
figure(8);
subplot(2,2,1),imshow(Image_noise_gauss),title('高斯噪声图像');
subplot(2,2,2),imshow(Image_noise_gauss_),title('高斯噪声图像');

%均值滤波
img_6=avg_filter(Image_noise_gauss,3 );
img_7=avg_filter(Image_noise_gauss_,3 );
subplot(2,2,3),imshow(img_6),title('均值滤波去噪图像')
subplot(2,2,4),imshow(img_7),title('均值滤波去噪图像')
 
% 加椒盐噪声
Image_noise_salt = imnoise(img_1,'salt & pepper'); %加噪
Image_noise_salt_ = imnoise(img_2,'salt & pepper'); %加噪
figure(9); 
subplot(2,2,1),imshow(Image_noise_salt),title('椒盐噪声图像');
subplot(2,2,2),imshow(Image_noise_salt_),title('椒盐噪声图像');

%中值滤波
img_8=median_filter(Image_noise_salt, 3 );
img_9=median_filter(Image_noise_salt_, 3 );
subplot(2,2,3),imshow(img_8),title('中值滤波去噪图像')
subplot(2,2,4),imshow(img_9),title('中值滤波去噪图像')
 



