clear all;

%% arguments
% anthropometrics (cm)
head_width = 15;
head_depth = 20;
shoulder_circumference = 140;
% pinnae resps on median plane
prtf_path = 'prtf.mat';
% noPinae resps path
noPinnae_path = '/Users/miccio/OneDrive - Aalborg Universitet/viking_measurements/viking2/4_sofa/subj_Z.sofa';
% output infos
indiv_subj = 'X';
indiv_sofa_path = 'indiv.sofa';

%% gather data
% calculate delays
[ITD, delays, delays_pos] = select_ITD(head_width, head_depth, shoulder_circumference, false);
% load PRTFs and their positions
load(prtf_path)
prtf = synthesized_hrtf;
prtf_pos = pos;
% load noPinnae hrirs 
SOFAstart;
obj = SOFAload(noPinnae_path);
np_hrirs = obj.Data.IR;
np_pos = obj.SourcePosition;
% sort rows into [az, el]
prtf_pos = prtf_pos(:, [2 1]);
prtf_pos(:,1) = mod(prtf_pos(:,1), 360);

%% decimate elevation on noPinnae (el_step = 10)
idx = find(mod(np_pos(:,2), 10) == 0);
np_pos = np_pos(idx,:);
np_hrirs = np_hrirs(idx,:,:);

%% interpolate delays
interp_delays = interpolateDelays(delays, delays_pos, np_pos);

%% generate output
indiv_hrirs = individualize_hrtf_nopinnae(prtf, prtf_pos, np_hrirs, np_pos, interp_delays);

%% store everything
head_width_m = head_width / 100;
save('data_for_sofa.mat', 'indiv_sofa_path', 'indiv_hrirs', 'np_pos', 'indiv_subj', 'head_width_m')

%% store as sofa
clear all;
load('data_for_sofa.mat')
generateIndividualizedSOFA(indiv_sofa_path, indiv_hrirs, np_pos, indiv_subj, head_width_m);

%% visualize some stuff
obj = SOFAload(indiv_sofa_path);
figure(); 
subplot(121); SOFAplotHRTF(obj, 'MagMedian', 1);
subplot(122); SOFAplotHRTF(obj, 'MagHorizontal', 1);
figure(); 
subplot(121); SOFAplotHRTF(obj, 'MagSagittal', 1, 'offset', 45);
subplot(122); SOFAplotHRTF(obj, 'MagSagittal', 2, 'offset', 45);
