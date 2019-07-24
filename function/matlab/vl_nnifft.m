function y = vl_nnifft(x, param, dzdy)

if nargin == 2
    
     y = ifft2(fftshift(fftshift(complex(x(:,:,1:end/2,:),x(:,:,end/2+1:end,:)),1),2));  %3T learning
     
   
    if param.isreal
        y = real(y);    
   else
        y = cat(3, real(y), imag(y));
        
    end
    
else
    if param.isreal
        
        y = fftshift(fftshift(fft2(ifftshift(ifftshift(dzdy, 1), 2)), 1), 2);

    else
        
        y = fftshift(fftshift(fft2(complex(dzdy(:,:,1:end/2,:),dzdy(:,:,end/2+1:end,:))),1),2);   %3T learning
        
    end
    
    y = cat(3, real(y), imag(y));
    
end