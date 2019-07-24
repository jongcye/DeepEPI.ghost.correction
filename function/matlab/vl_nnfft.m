function y = vl_nnfft(x, param, dzdy)

if nargin == 2
    if param.isreal
        y = fft2(x);
    else
        y = fft2(complex(x(:,:,1:end/2,:), x(:,:,end/2+1:end,:)));
    end
    
    y = cat(3, real(y), imag(y));
else
    dzdy = complex(dzdy(:,:,1:end/2,:), dzdy(:,:,end/2+1:end,:));
    y = ifft2(dzdy);
    
    if param.isreal
        y = real(y);
    else
        y = cat(3, real(y), imag(y));
    end
end