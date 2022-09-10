%%%% CST fit airfoil
%%%%拟合公式    y/c=C(x/c)S(x/c)+(x/c)*Zte/c
%%%%%%%%%%%%    C(x/c)=(x/c)^0.5*(1-x/c)
%%%%%%%%%%%%%   S(x/c)=sum(bi*n!/i!*(n-i)!*(x/c)^i*(1-x/c)^(n-i))
%N:每个表面的基函数个数（还有常数项所以实际上输入维度+1）
%Nu：上表面点的数目
%翼型通过dat读取
%案例：[mse,absError,maxError,b0]=CSTfoil(6,130)
function [mse,absError,maxError,b0]=CST_shapetoparam(N,Nu,filename)
% N=5;%拟合上表面分别需要的参数
N1=N;%拟合下表面需要的参数个数
% fid=fopen('RAE2822.dat','r');
% dirstr='results_RAE';
fid=fopen(filename,'r');
dirstr='CST_NACA';
c=textscan(fid,'%f%f');
M=cell2mat(c);
 fclose(fid);
% Nu=121;  %这个是REA2822的上表面数目
% Nu=130;  %这个是NACA0012的上表面点的数
xu=M(1:Nu,1);
yu=M(1:Nu,2);
xl=M(Nu+1:end,1);
yl=M(Nu+1:end,2);
% subplot(2,1,1)
% plot(xu,yu,xl,yl)
a=zeros(Nu,N+1);
if xu(end)>xl(end)
    Zte=yu(end)/xu(end);
else 
    Zte=yl(end)/xl(end);
end
yuu=yu-xu*Zte;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%翼型上表面拟合%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:Nu %翼型边界上点的个数
     for j=1:1:N+1
     a(i,j)=sqrt(xu(i))*(1-xu(i))*(factorial(N)/(factorial(j-1)*factorial(N-j+1)))*xu(i)^(j-1)*(1-xu(i))^(N-j+1);
     end
end 
   b=inv(a'*a)*a'*yuu;%b是获得的多项式系数
   b=b';
   k=1;
  for  x=0:0.001:1
     for j=1:1:N+1%根据求得的多项式系数拟合出新的翼型坐标
     s(j)=b(j)*(factorial(N)/(factorial(j-1)*factorial(N-j+1)))*x^(j-1)*(1-x)^(N-j+1);
     end
     %plot(x,s(j))
     y_cst(k)=sqrt(x)*(1-x)*sum(s)+x*Zte;
     k=k+1;
  end
  %%%%%%%%%%%%%%%%%%%%%拟合误差分析%%%%%%%%%%%%%%
  k=1;
for i=1:1:Nu
     for j=1:1:N+1
     s(j)=b(j)*(factorial(N)/(factorial(j-1)*factorial(N-j+1)))*xu(i)^(j-1)*(1-xu(i))^(N-j+1);
     end
     yyu(k,1)=sqrt(xu(i))*(1-xu(i))*sum(s)+xu(i)*Zte;
     k=k+1;
end 
   ggu=yu-yyu;%yu是原翼型的坐标，yyu是拟合之后的坐标
  gu=sum((yu-yyu).^2);
 %%%%%%%%%%%%%%%%%%%%%%%翼型下表面CST拟合%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Nl=length(xl); 
 al=zeros(Nl,N1+1);
  Zlte=-0;
 yll=yl-xl*Zlte;
for i=1:1:Nl
     for j=1:1:N1+1
     al(i,j)=sqrt(xl(i))*(1-xl(i))*(factorial(N1)/(factorial(j-1)*factorial(N1-j+1)))*xl(i)^(j-1)*(1-xl(i))^(N1-j+1);
     end
end 
   bl=inv(al'*al)*al'*yll;
   bl=bl';
   k=1;
   b0=[b,bl];
  for  x=0:0.001:1;
     for j=1:1:N1+1
     sl(j)=bl(j)*(factorial(N1)/(factorial(j-1)*factorial(N1-j+1)))*x^(j-1)*(1-x)^(N1-j+1);
     end
     y_cstl(k)=sqrt(x)*(1-x)*sum(sl)+x*Zlte;
     k=k+1;
  end 
%  x0=1:-0.001:0;
 x=0:0.001:1;
%  x=[x0,x];
%  yy_cst=[y_cst,y_cstl];

%  plot(x,y_cst,x,y_cstl);
 x=x';
  y_cst=y_cst';
    y_cstl=y_cstl';
%  M=[x,y_cst,zeros(1001,1);x,y_cstl,zeros(1001,1)]
xxx=[xu;xl];
yyy=[yu;yl];
%  plot(xxx,yyy,'g')

    %%%%%%%%%%%%%%%%%%%%%拟合误差分析%%%%%%%%%%%%%%
  k=1;
for i=1:1:Nl
     for j=1:1:N+1
     sl(j)=bl(j)*(factorial(N)/(factorial(j-1)*factorial(N-j+1)))*xl(i)^(j-1)*(1-xl(i))^(N-j+1);
     end
     yyl(k,1)=sqrt(xl(i))*(1-xl(i))*sum(sl)+xl(i)*Zlte;
     k=k+1;
end 
     ggl=yl-yyl;
  gl=sum((yl-yyl).^2) ;
%   subplot(2,1,2)
%   hold on
    xx=[-xl;xu];
    gg=[ggl;ggu];
%      plot(xx,gg);
x_origin=xxx;
x_nihe=xx;
y_nihe=[yyu;yyl];
y_origin=yyy;
g_nihewucha=gg;
nihe=[x_origin,y_origin,y_nihe];
nihewucha=[x_nihe,g_nihewucha];
% save fitted aifoil
% str1=[dirstr,'\nihe_CST,N=',num2str(N),'.dat'];
% str2=[dirstr,'\nihewucha_CST,N=',num2str(N),'.dat'];
% save(str1,'nihe','-ascii');%保存翼型拟合;
% save(str2,'nihewucha','-ascii');
mse=(gu+gl)/length(M);
absError=(sum(abs(ggu))+sum(abs(ggl)))/length(M);
maxError=max([max(ggu),max(ggl)]);