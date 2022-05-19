function [J_Erosion] = corrode(I,B)
n=size(B,1);
ind=B==0;
n_l=floor(n/2);
%对边界图进行扩充，目的是为了处理边界点
I_pad=padarray(I,[n_l,n_l],'replicate');
[M,N]=size(I);
J_Erosion=zeros(M,N);
for i=1:M
    for j=1:N
        %获得图像子块区域
        Block=I_pad(i:i+2*n_l,j:j+2*n_l);
        C=Block.*B;
        %删除0值，保留4连通数值
        C=C(:);
        C(ind)=[];
        %腐蚀操作
        J_Erosion(i,j)=min(C);
    end
end
end

