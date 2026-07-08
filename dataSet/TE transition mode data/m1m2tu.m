clc,clear all
load M1M2XMEAS
load M1M2XMV
X=simout;
plot(simout(1:1200,[2 3 7]));
xlabel('Samples');
ylabel('The values of three variables');
legend('XMEAS2','XMEAS3','XMEAS7');
set(gca,'FontSize',15);