function generateHrtfSet(head_width, head_depth, shoulder_circumference, prtf_path, indiv_sofa_path)
    % noPinae resps path
    pinnaless_path = 'data/subj_Z.sofa';
    % sofa subject name
    indiv_subj = 'X';

    %% gather data
    % calculate delays
    [~, delays, delays_pos] = select_ITD(head_width, head_depth, shoulder_circumference, false);
    % load PRTFs and their positions
    load(prtf_path, 'synthesized_hrtf', 'pos')
    prtf = synthesized_hrtf;
    prtf_pos = pos;
    % load pinnaless hrirs 
    SOFAstart;
    obj = SOFAload(pinnaless_path);
    np_hrirs = obj.Data.IR;
    np_pos = obj.SourcePosition;
    % sort rows into [az, el]
    prtf_pos = prtf_pos(:, [2 1]);
    prtf_pos(:,1) = mod(prtf_pos(:,1), 360);

    %% decimate elevation on pinnaless data (el_step = 10)
    idx = find(mod(np_pos(:,2), 10) == 0);
    np_pos = np_pos(idx,:);
    np_hrirs = np_hrirs(idx,:,:);

    %% interpolate delays
    interp_delays = interpolateDelays(delays, delays_pos, np_pos);

    %% generate output
    indiv_hrirs = individualizeHrtf(prtf, prtf_pos, np_hrirs, np_pos, interp_delays);

    %% head width in meters
    head_width_m = head_width / 100;

    %% store as sofa
    generateIndividualizedSOFA(indiv_sofa_path, indiv_hrirs, np_pos, indiv_subj, head_width_m);

    %% visualize some stuff
    obj = SOFAload(indiv_sofa_path);
    figure(); 
    subplot(121); SOFAplotHRTF(obj, 'MagMedian', 1);
    subplot(122); SOFAplotHRTF(obj, 'MagHorizontal', 1);
    figure(); 
    subplot(121); SOFAplotHRTF(obj, 'MagSagittal', 1, 'offset', 45);
    subplot(122); SOFAplotHRTF(obj, 'MagSagittal', 2, 'offset', 45);

end