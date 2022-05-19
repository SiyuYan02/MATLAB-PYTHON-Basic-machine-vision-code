
%% 获取外参数及M
aw=[1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1];
ai = [2240.4 0 595 0;0 2234.4 434.8 0;0 0 1 0];
bw = [0.9996 -0.0155 0.0221 -58.7961;0.0158 0.9998 -0.0156 21.0311;-0.0218 0.0159 0.9996 16.8578;0 0 0 1];
bi = [2209 0 578.7 0;0 2206.1 462.7 0;0 0 1 0];
a = ai*aw;
b = bi*bw;

%% 3D
file_path1 ='D:\temp\homework_data\fish1\left1\';% 图像文件夹路径
img_path_list1 = dir(strcat(file_path1,'*.jpg'));
file_path2 ='D:\temp\homework_data\fish1\right1\';% 图像文件夹路径
img_path_list2 = dir(strcat(file_path2,'*.jpg'));

img_num = length(img_path_list1);%获取图像总数量
if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        image_name1 = img_path_list1(j).name;% 图像名
        image_name2 = img_path_list2(j).name;% 图像名
        image1 =imread(strcat(file_path1,image_name1));
        image2 =imread(strcat(file_path2,image_name2));
        [u1,v1]=FindFish(image1);
        [u2,v2]=FindFish(image2);
        [x,y,z]=get_3D(a,b,u2,v2,u1);
        q(j)=x;
        w(j)=y;
        e(j)=z;
     end
end
plot3(q,w,e);

