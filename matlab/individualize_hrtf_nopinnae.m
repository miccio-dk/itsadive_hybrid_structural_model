function hrirs = individualize_hrtf_nopinnae(prtf, prtf_pos, np_hrirs, np_pos, delays)
    nfft = 512;
    fs = 48000;
    N = 60;
    % allocate new hrirs according to nP grid
    hrirs = NaN(size(np_hrirs));
    % get all elevations in the nP grid
    np_elevations = unique(np_pos(:,2));    
    
    % for each nP elevation...
    for el = np_elevations'
        % generate yulewalk filters for current front prtf
        curr_prtf_f = prtf((prtf_pos(:,2) == el) & (prtf_pos(:,1) == 0), :);
        curr_prtf_lin_f = db2mag(curr_prtf_f);
        [b_f, a_f] = yulewalk(N, linspace(0, 1, 257), curr_prtf_lin_f);
        % generate yulewalk filters for current back prtf
        curr_prtf_b = prtf((prtf_pos(:,2) == el) & (abs(prtf_pos(:,1)) == 180), :);
        curr_prtf_lin_b = db2mag(curr_prtf_b);
        [b_b, a_b] = yulewalk(N, linspace(0, 1, 257), curr_prtf_lin_b);
        
        % extract indexes for current nP elevation
        curr_idx = find(np_pos(:,2) == el);
        % for each index of current nP...
        for idx = curr_idx'
            % get current azimuth
            az = np_pos(idx,1);
            % get current nP
            np_hrir_curr = squeeze(np_hrirs(idx,:,:));
            % filter it with correct [b,a] for front/back
            if az > 90 & az < 270
                hrir_filt = filter(b_b, a_b, np_hrir_curr);
            else
                hrir_filt = filter(b_f, a_f, np_hrir_curr);
            end
            % store it
            hrirs(idx,:,:) = hrir_filt;
        end
        
    end
    
    % estimate delays current hrir
    [~, hrirs_filt_delays] = itdestimator(hrirs, 'fs', fs);
    % calcualte delay difference in samples
    delays_adj = round((delays - hrirs_filt_delays) * fs);
    % apply delay adjustment
    for i = 1:size(hrirs, 1)
        hrirs(i,1,:) = circshift(hrirs(i,1,:), delays_adj(i,1,:));
        hrirs(i,2,:) = circshift(hrirs(i,2,:), delays_adj(i,2,:));
    end
end