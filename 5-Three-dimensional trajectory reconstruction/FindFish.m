function [u,v] = FindFish(I)
%定位斑马鱼
I=rgb2gray(I);
[m,n]=size(I);
count=0;
u=0;
v=0;
%为了消除鱼缸周围对阈值化结果的影响
%在遍历中舍弃了鱼纲周围部分，只对鱼缸本身部分进行遍历
for i=120:m-290
    for j=140:n-120
        if I(i,j)<50
            count=count+1;
            u=u+i;
            v=v+j;
        end
    end
end
u=u/count;
v=v/count;
end

