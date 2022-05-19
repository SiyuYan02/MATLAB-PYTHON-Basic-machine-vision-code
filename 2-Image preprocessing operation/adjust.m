function [imag_] = adjust(imag,a,b)
%a调整对比度，b调整亮度
[w, h] = size(imag);
for i = 1:w
    for j = 1:h
        imag_(i, j) = round(imag(i, j).*(1+a)+b);
        if (imag_(i, j) > 255)
            imag_(i, j) = 255;
        end
    end
end
end

