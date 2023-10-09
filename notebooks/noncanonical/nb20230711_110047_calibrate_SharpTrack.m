%% scr20230711_110047_calibrate_SharpTrack.mlx
% 
% 
% 
% 
% 
%%
% 
%                            name                             acronym      AP_location    DV_location    ML_location    avIndex
%      _________________________________________________    ___________    ___________    ___________    ___________    _______
%      {'Caudoputamen'                                 }    {'CP'     }        0.22          3.46           -2.82       {[574]}
%      {'Primary somatosensory area upper limb layer 1'}    {'SSp-ul1'}       -0.11          1.18           -2.82       {[ 81]}
%      {'Piriform-amygdalar area'                      }    {'PAA'    }       -0.34          7.37           -2.82       {[445]}
%
%% 
% 


shtr.AP = [-0.11, 0.22, -0.34]
shtr.DV = [1.18, 3.46, 7.37]

pafr.AP = [-0.284, 0.297, -0.370]
pafr.DV = [0.361, 2.429, 6.012]

%%
figure
ax = axes
plot(shtr.AP, shtr.DV, 'x', DisplayName='SharpTrack')
hold on
plot(pafr.AP, pafr.DV, 'o', DisplayName='Paxinos')
ax.YDir = 'reverse';
xlim([-3, 7])
ylim([-1. 9])
box off;
tickdir out;
legend(Location='northeast')
xlabel('AP (mm from bregma)')
ylabel('DV (mm from bregma)')
%%
figure;
ax = axes;
plot(shtr.DV, pafr.DV, 'o')
box off;
tickdir out;
xlabel('DV in SharpTrack (mm from bregma)')
ylabel('DV in Paxinos (mm from bregma)')

X = [ones(size(shtr.DV')) , shtr.DV']
Y = pafr.DV';
b = X\Y

x = -2:0.01:8; % 10 micron
y = b(1) + b(2) * x;

hold on
ax.XTick = -2:8;
plot(x, y);
plot(xlim, [0 0], 'k:')
plot([x(find(y >=0,1,"first")), x(find(y >=0,1,"first"))], ylim, 'k:')
%%
fprintf("At y = 0, i.e. the real bregma, SharpTrack DV value is %.3f mm\n", b(1));
fprintf("SharpTrack depth x can be converted to Paxinos depth by y = %.3f * x %+.3f\n", b(2), b(1));