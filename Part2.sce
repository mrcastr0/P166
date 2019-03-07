/////ANAMORPHIC PROPERTY OF FT//////////////////
tallrec = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\tall_rec_ap.png'); //load Tall rectangular aperture image
widerec = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\wide_rec_ap.png'); //load Wide rectangular aperture image
dots = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\2_dots.png'); //load 2 Dots image
dotsv2 = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\2_dots_.png');//load 2 Dots with smaller distance between image
dotsg = rgb2gray(dots); //Convert image into a M X N X 1 (monochromatic image)
dotsgv2 = rgb2gray(dotsv2);

tallrec_gray = fft2(double(tallrec)); //Convert the image into a double and apply FFT
tallrecgray_dp = mat2gray(abs(tallrec_gray)); //Convert matrix using a double precision transform
tallrecgrayshift_dp = mat2gray(abs(fftshift(tallrec_gray))); //Perform FFT shift to center the figure
imwrite(tallrecgray_dp,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTTallrec_dp.png');
imwrite(tallrecgrayshift_dp,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTShiftTallrec_dp.png');

widerec_gray = fft2(double(widerec));
widerecgray_dp = mat2gray(abs(widerec_gray));
widerecgrayshift_dp = mat2gray(abs(fftshift(widerec_gray)));
imwrite(widerecgray_dp,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTWiderec_dp.png');
imwrite(widerecgrayshift_dp,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTShiftWiderec_dp.png');

dots_gray = fft2(double(dotsg));
dotsgray_dp = mat2gray(abs(dots_gray));
dotsgrayshift_dp = mat2gray(abs(fftshift(dots_gray)));
imwrite(dotsgray_dp,'C:\Users\Marc Castro\Desktop\p166\Part2\FFT2dots_dp.png');
imwrite(dotsgrayshift_dp,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTShift2dots_dp.png');


dotsv2_gray = fft2(double(dotsgv2));
dotsv2gray_dp = mat2gray(abs(dotsv2_gray));
dotsv2grayshift_dp = mat2gray(abs(fftshift(dotsv2_gray)));
imwrite(dotsv2gray_dp,'C:\Users\Marc Castro\Desktop\p166\Part2\FFT2dotsv2_dp.png');
imwrite(dotsv2grayshift_dp,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTShift2dotsv2_dp.png');

//////////ROTATION OF FT//////////////
nx = 128
ny = 128
x = linspace(-1,1,nx);
y = linspace(-1,1,ny);
[Y,X] = ndgrid(x,y);

f = 5;
z = sin(2*%pi*f*X);
imwrite(z,'C:\Users\Marc Castro\Desktop\p166\Part2\roof.png');
roof_load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\roof.png');
roof_gray = fft2(double(roof_load));
FFT_roof = mat2gray(abs(roof_gray));
imwrite(FFT_roof,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTroof_dp.png');
FFTshift_roof = mat2gray(abs(fftshift(roof_gray)));
imwrite(FFTshift_roof,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTshiftroof_dp.png');

f = 10;
z = sin(2*%pi*f*X);
imwrite(z,'C:\Users\Marc Castro\Desktop\p166\Part2\roof_f10.png');
roof_f10load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\roof_f10.png');
roof_f10gray = fft2(double(roof_f10load));
FFT_rooff10 = mat2gray(abs(roof_f10gray));
imwrite(FFT_rooff10,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTrooff10_dp.png');
FFTshift_rooff10 = mat2gray(abs(fftshift(roof_f10gray)));
imwrite(FFTshift_rooff10,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTshiftrooff10_dp.png');

f = 15;
z = sin(2*%pi*f*X);
imwrite(z,'C:\Users\Marc Castro\Desktop\p166\Part2\roof_f15.png');
roof_f15load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\roof_f15.png');
roof_f15gray = fft2(double(roof_f15load));
FFT_rooff15 = mat2gray(abs(roof_f15gray));
imwrite(FFT_rooff15,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTrooff15_dp.png');
FFTshift_rooff15 = mat2gray(abs(fftshift(roof_f15gray)));
imwrite(FFTshift_rooff15,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTshiftrooff15_dp.png');

f = 5;
z = sin(2*%pi*f*X) + 1;
imwrite(z,'C:\Users\Marc Castro\Desktop\p166\Part2\roof_f15+1.png');
roof_f15load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\roof_f15+1.png');
roof_f15gray = fft2(double(roof_f15load));
FFT_rooff15 = mat2gray(abs(roof_f15gray));
imwrite(FFT_rooff15,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTrooff15+1_dp.png');
FFTshift_rooff15 = mat2gray(abs(fftshift(roof_f15gray)));
imwrite(FFTshift_rooff15,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTshiftrooff15+1_dp.png');

f=10;
theta = 30;
z1 = sin(2*%pi*f*(X*sin(theta) + Y*cos(theta)));
imwrite(z1,'C:\Users\Marc Castro\Desktop\p166\Part2\roofrotated.png');
roofrotated_load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\roofrotated.png');
roofrotated_gray = fft2(double(roofrotated_load));
FFT_roofrotated = mat2gray(abs(roofrotated_gray));
imwrite(FFT_roofrotated,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTroofrotated_dp.png');
FFTshift_roofrotated = mat2gray(abs(fftshift(roofrotated_gray)));
imwrite(FFTshift_roofrotated,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTshiftroofrotated_dp.png');

z2 = sin(2*%pi*4*Y)*sin(2*%pi*4*X);
imwrite(z2,'C:\Users\Marc Castro\Desktop\p166\Part2\roofXYWave.png');
roofXYWave_load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\roofXYWave.png');
roofXYWave_gray = fft2(double(roofXYWave_load));
FFT_roofXYWave = mat2gray(abs(roofXYWave_gray));
imwrite(FFT_roofXYWave,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTroofXYWave_dp.png');
FFTshift_roofXYWave = mat2gray(abs(fftshift(roofXYWave_gray)));
imwrite(FFTshift_roofXYWave,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTshiftroofXYWave_dp.png');

z3 = sin(2*%pi*4*X)*sin(2*%pi*4*Y) * sin(2*%pi*f*(Y*sin(theta) + X*cos(60)));
imwrite(z3,'C:\Users\Marc Castro\Desktop\p166\Part2\roofmultWave.png');
roofmultWave_load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\roofmultWave.png');
roofmultWave_gray = fft2(double(roofmultWave_load));
FFT_roofmultWave = mat2gray(abs(roofmultWave_gray));
imwrite(FFT_roofmultWave,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTroofmultWave_dp.png');
FFTshift_roofmultWave = mat2gray(abs(fftshift(roofmultWave_gray)));
imwrite(FFTshift_roofmultWave,'C:\Users\Marc Castro\Desktop\p166\Part2\FFTshiftroofmultWave_dp.png');

///Convolution Theorem Redux
nx = 128; ny = 128;
x = linspace(-1,1,nx);
y = linspace(-1,1,ny);
[Y,X] = ndgrid(x,y);

///Create Dots
dots = zeros(nx,ny);
dist = 1/5;
to_center = dist*nx;
dots(64,64+to_center) = 1;
dots(64,64-to_center) = 1;
imwrite(dots, 'C:\Users\Marc Castro\Desktop\p166\Part2\2pxDots.png');
FFT_dots = fftshift(mat2gray(abs(fft2(dots))));
imwrite(FFT_dots, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTDots.png');

////Create Circles
nx2 =32;ny2=32;
x2 = linspace(-1,1,nx2);
y2 = linspace(-1,1,ny2);
[Y2,X2] = ndgrid(x2,y2);

r2 = sqrt(X2.^2 + Y2.^2);
circ = zeros(nx2,ny2);
circ(find(r2<0.2)) =1;
circ2=zeros(nx,ny);
circ2(48:79 , 80:111) = circ;
circ2(48:79 , 16:47) = circ;
imwrite(circ2, 'C:\Users\Marc Castro\Desktop\p166\Part2\2Circles.png');
circleFT = fftshift(mat2gray(abs(fft2(circ2))));
imwrite(circleFT, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTCircle.png');

///Create Squares
sq =zeros(nx2,ny2);
sq(find(abs(X2)<0.25 & abs(Y2)<0.25)) = 1;
sq2=zeros(nx,ny);
sq2(48:79,16:47) = sq;
sq2(48:79,80:111) = sq;
imwrite(sq2, 'C:\Users\Marc Castro\Desktop\p166\Part2\2Square.png');
squareFT = fftshift(mat2gray(abs(fft2(sq2))));
imwrite(squareFT, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTSquare.png');

///Create Gaussians
sigma = 0.2;
mu = 0;
amp = 1/(sigma*sqrt(2*%pi));
x3 = linspace(-5,5,nx);
y3 = linspace(-5,5,ny);
gaus2 = zeros(nx,ny);
[Y3,X3] = ndgrid(x3,y3);
r = sqrt(X3.^2 + Y3.^2);

gausR = r(:,dist*nx+1:nx/2+dist*nx);
gausL = r(:,nx/2-dist*nx+1:nx-dist*nx);
gau = amp*exp(-(([gausR,gausL]-mu).^2)/(2*sigma.^2));
imwrite(gau, 'C:\Users\Marc Castro\Desktop\p166\Part2\2Gaus.png');
gausFT = fftshift(mat2gray(abs(fft2(gau))));
imwrite(gausFT, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTgaus.png');

sigma = 0.5;
mu = 0;
amp = 1/(sigma*sqrt(2*%pi));
x3 = linspace(-5,5,nx);
y3 = linspace(-5,5,ny);
gaus2 = zeros(nx,ny);
[Y3,X3] = ndgrid(x3,y3);
r = sqrt(X3.^2 + Y3.^2);

gausR = r(:,dist*nx+1:nx/2+dist*nx);
gausL = r(:,nx/2-dist*nx+1:nx-dist*nx);
gau = amp*exp(-(([gausR,gausL]-mu).^2)/(2*sigma.^2));
imwrite(gau, 'C:\Users\Marc Castro\Desktop\p166\Part2\2Gausv2.png');
gausFT = fftshift(mat2gray(abs(fft2(gau))));
imwrite(gausFT, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTgaus2.png');


sigma = 0.7;
mu = 0;
amp = 1/(sigma*sqrt(2*%pi));
x3 = linspace(-5,5,nx);
y3 = linspace(-5,5,ny);
gaus2 = zeros(nx,ny);
[Y3,X3] = ndgrid(x3,y3);
r = sqrt(X3.^2 + Y3.^2);

gausR = r(:,dist*nx+1:nx/2+dist*nx);
gausL = r(:,nx/2-dist*nx+1:nx-dist*nx);
gau = amp*exp(-(([gausR,gausL]-mu).^2)/(2*sigma.^2));
imwrite(gau, 'C:\Users\Marc Castro\Desktop\p166\Part2\2Gausv3.png');
gausFT = fftshift(mat2gray(abs(fft2(gau))));
imwrite(gausFT, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTgaus3.png');

n = 200;
pattern = [-1 1 -1 2 -1 2 -1 1 -1
           -1 1 -1 2 -1 2 -1 1 -1
           -1 1 -1 2 -1 2 1 1 -1
           -1 1 -1 2 -1 -2 -1 1 -1
           -1 -1 -1 -2 1 2 -1 1 -1
           -1 1 -1 2 -1 2 -1 1 -1
           -1 1 1 -2 -1 2 -1 1 -1
           -1 1 -1 2 -1 2 -1 1 -1
           -1 1 -1 2 -1 2 -1 1 -1];          
d = zeros(n,n);
d(96:104,96:104) = pattern;
A= imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\10Points.png');
FFTA = fft2(double(A));
FFTA_out = mat2gray(abs(FFTA));
imwrite(FFTA_out,'C:\Users\Marc Castro\Desktop\p166\Part2\FFT10p.png');
FFTd = fftshift(d);
convAD = FFTA_out.*FFTd;
OUTPUT = mat2gray(abs(fft2(convAD)));
imwrite(OUTPUT, 'C:\Users\Marc Castro\Desktop\p166\Part2\convAD.png');


d2 = zeros(n,n);
d2(25,25) = 1; d2(50,25) = 1; d2(75,25) = 1; d2(100,25) = 1; d2(125,25) = 1; d2(150,25) = 1; d2(175,25) = 1;
d2(25,50) = 1; d2(50,50) = 1; d2(75,50) = 1; d2(100,50) = 1; d2(125,50) = 1; d2(150,50) = 1; d2(175,50) = 1;
d2(25,75) = 1; d2(50,75) = 1; d2(75,75) = 1; d2(100,75) = 1; d2(125,75) = 1; d2(150,75) = 1; d2(175,75) = 1;
d2(25,100) = 1; d2(50,100) = 1; d2(75,100) = 1; d2(100,100) = 1; d2(125,100) = 1; d2(150,100) = 1; d2(175,100) = 1;
d2(25,125) = 1; d2(50,125) = 1; d2(75,125) = 1; d2(100,125) = 1; d2(125,125) = 1; d2(150,125) = 1; d2(175,125) = 1; 
d2(25,150) = 1; d2(50,150) = 1; d2(75,150) = 1; d2(100,150) = 1; d2(125,150) = 1; d2(150,150) = 1; d2(175,150) = 1;
d2(25,175) = 1; d2(50,175) = 1; d2(75,175) = 1; d2(100,175) = 1; d2(125,175) = 1; d2(150,175) = 1; d2(175,175) = 1;
imwrite(d2, 'C:\Users\Marc Castro\Desktop\p166\Part2\equal25points.png');
FFTd2 = mat2gray(abs(fftshift(fft2(d2))));
imwrite(FFTd2, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTd25.png');


d3 = zeros(n,n);
d3(30,45) = 1; d3(50,45) = 1; d3(75,45) = 1; d3(100,45) = 1; d3(125,45) = 1; d3(150,45) = 1; d3(170,45) = 1;
d3(25,50) = 1; d3(50,50) = 1; d3(75,50) = 1; d3(100,50) = 1; d3(125,50) = 1; d3(150,50) = 1; d3(175,50) = 1;
d3(30,85) = 1; d3(50,85) = 1; d3(75,85) = 1; d3(100,85) = 1; d3(125,85) = 1; d3(150,85) = 1; d3(170,85) = 1;
d3(25,100) = 1; d3(50,100) = 1; d3(75,100) = 1; d3(100,100) = 1; d3(125,100) = 1; d3(150,100) = 1; d3(175,100) = 1;
d3(30,110) = 1; d3(50,110) = 1; d3(75,110) = 1; d3(100,110) = 1; d3(125,110) = 1; d3(150,110) = 1; d3(170,110) = 1; 
d3(25,150) = 1; d3(50,150) = 1; d3(75,150) = 1; d3(100,150) = 1; d3(125,150) = 1; d3(150,150) = 1; d3(175,150) = 1;
d3(30,170) = 1; d3(50,170) = 1; d3(75,170) = 1; d3(100,170) = 1; d3(125,170) = 1; d3(150,170) = 1; d3(170,170) = 1;
imwrite(d3, 'C:\Users\Marc Castro\Desktop\p166\Part2\equal25v2points.png');
FFTd3 = mat2gray(abs(fftshift(fft2(d3))));
imwrite(FFTd3, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTd35.png');




/////FingerPrint
finger_load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\fingerprint.jpg');
finger_gray =rgb2gray(finger_load);
finger = mat2gray(double(finger_gray));
imwrite(finger, 'C:\Users\Marc Castro\Desktop\p166\Part2\Finger.png');
FFTfinger = mat2gray(log(abs(fftshift(fft2(finger)))));
imwrite(FFTfinger, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTFinger.png');
filt = FFTfinger;
filt(find(FFTfinger<0.675)) = 0;
imwrite(filt, 'C:\Users\Marc Castro\Desktop\p166\Part2\Filter_finger.png');
enhance = mat2gray(abs(fft2(fft2(finger).*fftshift(filt))));
enhanced = imrotate(enhance,180);
imwrite(enhanced, 'C:\Users\Marc Castro\Desktop\p166\Part2\Finger_enhanced.png');

filter_enhance = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\Filter_enhance.png');
filter_enhanceg = rgb2gray(filter_enhance);
filter_enhanced = mat2gray(double(filter_enhanceg));
enhancev2 = mat2gray(abs(fft2(fft2(finger).*fftshift(filter_enhanced))));
enhancedv2 = imrotate(enhancev2,180);
imwrite(enhancedv2, 'C:\Users\Marc Castro\Desktop\p166\Part2\Finger_enhancedv2.png');

///Moon
moon_load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\moon.jpg');
moon = mat2gray(double(moon_load));
imwrite(moon, 'C:\Users\Marc Castro\Desktop\p166\Part2\moongray.png');
FFTmoon = mat2gray(log(abs(fftshift(fft2(moon)))));
imwrite(FFTmoon, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTmoon.png');
filtermoon = FFTmoon;
filtermoon(find(FFTmoon<0.5)) = 1;
filtermoon(find(FFTmoon>0.5)) = 0;
filtermoon(282:292,269:279) = 1;
imwrite(filtermoon, 'C:\Users\Marc Castro\Desktop\p166\Part2\Filter_moon.png');
enhance = mat2gray(abs(fft2(fft2(moon).*(filtermoon))));
enhanced = imrotate(enhance,180);
imwrite(enhanced, 'C:\Users\Marc Castro\Desktop\p166\Part2\moon_enhanced.png');

filtermoon(:,:) = 1;
filtermoon(:,273:275) = 0;
filtermoon(286:288,:) = 0;
filtermoon(282:292,269:279) = 1;
imwrite(filtermoon, 'C:\Users\Marc Castro\Desktop\p166\Part2\Filter_moon2.png');
enhance = mat2gray(abs(fft2(fft2(moon).*fftshift(filtermoon))));
enhanced = imrotate(enhance,180);
imwrite(enhanced, 'C:\Users\Marc Castro\Desktop\p166\Part2\moon_enhanced2.png');

//Canvas
canvas_load = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Input\canvasweave.jpg');
canvag = rgb2gray(canvas_load);
canvas = mat2gray(double(canvag));
imwrite(canvas, 'C:\Users\Marc Castro\Desktop\p166\Part2\canvas.png');
FFTcanvas = mat2gray(log(abs(fftshift(fft2(canvas)))));
imwrite(FFTcanvas, 'C:\Users\Marc Castro\Desktop\p166\Part2\FFTcanvas.png');
filtercanvas = FFTcanvas;
filtercanvas(find(FFTcanvas<0.55)) = 1;
filtercanvas(find(FFTcanvas>0.55)) = 0;
filtercanvas(204.5:213.5,285.5:295.5) = 1;
imwrite(filtercanvas, 'C:\Users\Marc Castro\Desktop\p166\Part2\Filter_canvas.png');
enhance = mat2gray(abs(fft2(fft2(canvas).*(filtercanvas))));
enhanced = imrotate(enhance,180);
imwrite(enhanced, 'C:\Users\Marc Castro\Desktop\p166\Part2\canvas_enhanced.png');

Fcanv2 = imread('C:\Users\Marc Castro\Desktop\p166\Part2\Filter_canvas2.png');
Fcanvg = rgb2gray(Fcanv2);
Fcanvas = mat2gray(double(Fcanvg));
enhance = mat2gray(abs(fft2(fft2(canvas).*fftshift(Fcanvas))));
enhanced = imrotate(enhance,180);
imwrite(enhanced, 'C:\Users\Marc Castro\Desktop\p166\Part2\canvas_enhanced2.png');








