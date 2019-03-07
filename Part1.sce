//Circle
n = 128;
x = linspace(-1,1,n);
y = linspace(-1,1,n);
[X,Y] = ndgrid(x,y);
r = sqrt(X.^2 + Y.^2);
circle = zeros(n,n);
circle(find(r<0.3)) = 1;

imwrite(circle,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\circle.png');


//Circle FFT
circle_load = imread('C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\circle.png');
circle_gray = fft2(double(circle_load));
num3_dp = mat2gray(abs(circle_gray));
num4_dp = mat2gray(abs(fftshift(circle_gray)));
num5_dp = mat2gray(abs(fft2(circle_gray)));
num3_8bit = uint8(abs(circle_gray));
num4_8bit = uint8(abs(fftshift(circle_gray)));
num5_8bit = uint8(abs(fft2(circle_gray)));
rl = real(fftshift(circle_gray));
im = imag(fftshift(circle_gray));

//Circle Outputs
imwrite(num3_dp,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTCircle_dp.png');
imwrite(num4_dp,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTShiftCircle_dp.png');
imwrite(num5_dp,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTInvCircle_dp.png');
imwrite(num3_8bit,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTCircle_8bit.png');
imwrite(num4_8bit,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTShiftCircle_8bit.png');
imwrite(num5_8bit,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTInvCircle_8bit.png');
imwrite(rl,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\RLCircle.png');
imwrite(im,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\IMCircle.png');

//Letter A
A_x = imread('C:\Users\Marc Castro\Desktop\p166\Part1\A.png');
Agray = fft2(double(A_x));
num6a_8bit = uint8(abs(Agray));
num6b_8bit = uint8(abs(fftshift(Agray)));
num6c_8bit = uint8(abs(fft2(Agray)));
num6a_dp = mat2gray(abs(Agray));
num6b_dp = mat2gray(abs(fftshift(Agray)));
num6c_dp = mat2gray(abs(fft2(Agray)));
Arl_8bit = uint8(real(fftshift(Agray)));
Aim_8bit = uint8(imag(fftshift(Agray)));
Arl_dp = mat2gray(real(fftshift(Agray)));
Aim_dp = mat2gray(imag(fftshift(Agray)));


//A Outputs
imwrite(num6a_8bit,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTA_8bit.png');
imwrite(num6b_8bit,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTShiftA_8bit.png');
imwrite(num6c_8bit,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTInvA_8bit.png');
imwrite(num6a_dp,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTA_dp.png');
imwrite(num6b_dp,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTShiftA_dp.png');
imwrite(num6c_dp,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTInvA_dp.png');
imwrite(Arl_8bit,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\RLA_8bit.png');
imwrite(Aim_8bit,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\IMA_8bit.png');
imwrite(Arl_dp,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\RLA_dp.png');
imwrite(Aim_dp,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\IMA_dp.png');

//Corrugated Roof
nx = 128;
ny = 128;
xroof = linspace(-1,1,nx);
yroof = linspace(-1,1,ny);
[Xroof,Yroof] = ndgrid(xroof,yroof);
freq = 3;
z = sin(2*%pi*freq*Yroof);
imwrite(z,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\croof.png');
roof_load = imread('C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\croof.png');
roof_gray = fft2(double(roof_load));
num8a = mat2gray(abs(fftshift(roof_gray)));
num8av2 = uint8(abs(fftshift(roof_gray)));
imwrite(num8a,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTroof_dp.png');
imwrite(num8av2,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTroof_8bit.png');

//Double Slit
slit_load = imread('C:\Users\Marc Castro\Desktop\p166\Part1\double_slit.png');
slit_gray = fft2(double(slit_load));
num8b = mat2gray(abs(fftshift(slit_gray)));
num8bv2 = uint8(abs(fftshift(slit_gray)));
imwrite(num8b,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTslit_dp.png');
imwrite(num8bv2,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTslit_8bit.png');
	
//Square Function
z1 = squarewave(2 * %pi * 5 * Yroof);
imwrite(z1,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\squarewave.png');
square_load = imread('C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\squarewave.png');
square_gray = fft2(double(square_load));
num8c = mat2gray(abs(fftshift(square_gray)));
num8cv2 = uint8(abs(fftshift(square_gray)));
imwrite(num8c,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTsquare_dp.png');
imwrite(num8cv2,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTsquare_8bit.png');

//Gaussian Bell Curve
r = sqrt(Xroof.^2 + Yroof.^2);
A = (1/sqrt(2*%pi))*exp(-0.9*r/2);
B = zeros(nx,ny);
B(find(r<0.1)) = 1;
A = A.*B;
f = scf();
grayplot(xroof,yroof,A);
f.color_map = graycolormap(32);
xs2png(f,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\bcurve.png'); //Crop and resize using paint

bcurve_load = imread('C:\Users\Marc Castro\Desktop\p166\Part1\bcurve_parsed.png');
bgray = rgb2gray(bcurve_load);
bcurve_gray = fft2(double(bgray));
num8d = mat2gray(abs(bcurve_gray));
num8dv2 = uint8(abs(fftshift(bcurve_gray)));
num8dv3 = uint16(abs(fftshift(bcurve_gray)));
imwrite(num8d,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTbcurve_dp.png');
imwrite(num8dv2,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTbcurve_8bit.png');
imwrite(num8dv3,'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\FFTbcurve_16bit.png');

////////SIMULATION OF AN IMAGING DEVICE////////////////////////
//Small Circle
vip = imread('C:\Users\Marc Castro\Desktop\p166\Part1\VIP.png');
scircle = imread('C:\Users\Marc Castro\Desktop\p166\Part1\circle2.png');
fftshift_circ2 = fftshift(double(scircle));
fft_vip = fft2(double(vip));
FRA = fftshift_circ2.*fft_vip;
IRA = fft2(FRA);
FImage = mat2gray(abs(IRA));
imwrite(FImage, 'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\VIP_apertured.png');
//Circle
vcircle = imread('C:\Users\Marc Castro\Desktop\p166\Part1\circle.png');
fftshift_circv = fftshift(double(vcircle));
fft_vip = fft2(double(vip));
FRAv = fftshift_circv.*fft_vip;
IRAv = fft2(FRAv);
FImagev = mat2gray(abs(IRAv));
imwrite(FImagev, 'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\VIP_aperturedv.png');
//Bigger Circle
scircle2 = imread('C:\Users\Marc Castro\Desktop\p166\Part1\circle3.png');
fftshift_circ3 = fftshift(double(scircle2));
fft_vip2 = fft2(double(vip));
FRA2 = fftshift_circ3.*fft_vip;
IRA2 = fft2(FRA2);
FImage2 = mat2gray(abs(IRA2));
imwrite(FImage2, 'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\VIP_apertured2.png');


////////CORRELATION///////////////////////////////////
sent = imread('C:\Users\Marc Castro\Desktop\p166\Part1\sentence.png');
asmall = imread('C:\Users\Marc Castro\Desktop\p166\Part1\Asmall.png');
fft_sent = fft2(double(sent));
fft_asmall = fft2(double(asmall));
conj_sent = conj(fft_sent);
sentas = fft_asmall.*conj_sent;
inverse_sentas = mat2gray(abs(fftshift(fft2(sentas))));
imwrite(inverse_sentas, 'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\correlation.png');


/////////////EDGE DETECTION////////////////////////
n = 128;
raw_image = zeros(n,n);
//Image Augmentations
vertical = [-1 2 -1; -1 2 -1; -1 2 -1]
horizontal = [-1 -1 -1; 2 2 2;-1 -1 -1]
rightDiag = [-1 -1 2; -1 2 -1; 2 -1 -1]
leftDiag = [2 -1 -1; -1 2 -1; -1 -1 2]
spot = [-1 -1 -1;-1 8 -1; -1 -1 -1]

V = raw_image;
H = raw_image;
rD = raw_image;
lD = raw_image;
sp = raw_image;

V(63:65,63:65) = vertical;
H(63:65,63:65) = horizontal; 
rD(63:65,63:65) = rightDiag;
lD(63:65,63:65) = leftDiag;
sp(63:65,63:65) = spot;

vip = imread('C:\Users\Marc Castro\Desktop\p166\Part1\VIP.png');
vip_filter = conj(fft2(double(vip)));
vipfilter_gray = uint8(abs(vip_filter));
imwrite(vipfilter_gray, 'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\vip_filter.png');
V_fft = fft2(double(V));
H_fft = fft2(double(H));
rD_fft = fft2(double(rD));
lD_fft = fft2(double(lD));
sp_fft = fft2(double(sp));
conv_V = V_fft.*vip_filter;
conv_H = H_fft.*vip_filter;
conv_rD = rD_fft.*vip_filter;
conv_lD = lD_fft.*vip_filter;
conv_sp = sp_fft.*vip_filter;

imwrite(mat2gray(abs(fftshift(fft2(double(conv_V))))),'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\VIP_vertical_conv.png');
imwrite(mat2gray(abs(fftshift(fft2(double(conv_H))))),'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\VIP_horizontal_conv.png');
imwrite(mat2gray(abs(fftshift(fft2(double(conv_rD))))),'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\VIP_rightdiag_conv.png');
imwrite(mat2gray(abs(fftshift(fft2(double(conv_lD))))),'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\VIP_leftdiag_conv.png');
imwrite(mat2gray(abs(fftshift(fft2(double(conv_sp))))),'C:\Users\Marc Castro\Desktop\p166\Part1\Outputs\VIP_spot_conv.png');
