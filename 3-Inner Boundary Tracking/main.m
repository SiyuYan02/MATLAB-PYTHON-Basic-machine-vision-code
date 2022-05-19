I=imread('img.jpg');
figure(1);  
imshow(I);title('原图')
%图像灰度化处理
A = rgb2gray(I);
figure(2);  
imshow(A);
A_max=max(A);
A_min=min(A);
title('灰度图')
T =(A_max+A_min)/2 ;   %取均值作为初始阈值
done = false;   %定义跳出循环的量
i = 0;

% while循环进行迭代
while ~done
    r1 = find(A<=T);  %小于阈值的部分
    r2 = find(A>T);   %大于阈值的部分
    Tnew = (mean(A(r1)) + mean(A(r2))) / 2;  %计算分割后两部分的阈值均值的均值
    done = abs(Tnew - T) < 0.01;     %判断迭代是否收敛
    T = Tnew;      %如不收敛,则将分割后的均值的均值作为新的阈值进行循环计算
    i = i+1;
end
A(r1) = 0;   %将小于阈值的部分赋值为0
A(r2) = 1;   %将大于阈值的部分赋值为1   这两步是将图像转换成二值图像

figure(3);
imshow(A,[]);
title('迭代处理后')

%高斯滤波
var=0.5;
W = fspecial('gaussian',[5,5],var);
img= imfilter(A, W, 'replicate');
[m,n]=size(img);
for i=1:1:m*n
   if img(i)<0.9
      img(i)=0;
   else
      img(i)=1;
   end
end

figure(5);
imshow(img,[]);
title('滤波处理后')

%内边界处理
imgn=zeros(m,n);        %边界标记图像
ed=[-1 -1;0 -1;1 -1;1 0;1 1;0 1;-1 1;-1 0]; %从左上角像素，逆时针搜索
for i=2:m-1
    for j=2:n-1
        if img(i,j)==1 && imgn(i,j)==0      %当前是没标记的白色像素
            if sum(sum(img(i-1:i+1,j-1:j+1)))~=9    %块内部的白像素不标记
                ii=i;         %像素块内部搜寻使用的坐标
                jj=j;
                imgn(i,j)=2;    %本像素块第一个标记的边界，第一个边界像素为2
                
                while imgn(ii,jj)~=2    %是否沿着像素块搜寻一圈了。
                    for k=1:8           %逆时针八邻域搜索
                        tmpi=ii+ed(k,1);        %八邻域临时坐标
                        tmpj=jj+ed(k,2);
                        if img(tmpi,tmpj)==1 && imgn(tmpi,tmpj)~=2  %搜索到新边界，并且没有搜索一圈
                            ii=tmpi;        %更新内部搜寻坐标，继续搜索
                            jj=tmpj;
                            imgn(ii,jj)=1;  %边界标记图像该像素标记，普通边界为1
                            break;
                        end
                    end
                end
                
            end
        end
    end
end
 
figure(6);
imgn=imgn>=1;
imshow(imgn,[]);
