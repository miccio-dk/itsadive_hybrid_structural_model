function [ITD, delays, pos] = select_ITD(head_width, head_depth, shoulder_circumference, plots)
    close all
    amt_start
    warning('off','all');
    
    HUTUBS_path = '../hutubs/';
    
    load('anthro_HUTUBS.mat','anthro_HUTUBS');
    subjectId_HUTUBS = subject_num_list_HUTUBS;
    anthro_HUTUBS = anthro_HUTUBS(subjectId_HUTUBS,[1 3 13]); % HUTUBS relevant parameters
    anthro_HUTUBS = table2array(anthro_HUTUBS);
    
    sigma = [1.031 1.189 11.552]; % training set std values
    model_coeffs = [2.135 0.625 1.845]; % linear model coefficients
    
    anthro_ind = [head_width head_depth shoulder_circumference];
    scores = zeros(length(anthro_HUTUBS),1);
    for i = 1:length(anthro_HUTUBS)
        diffs = (anthro_ind - anthro_HUTUBS(i,:))./sigma;
        scores(i) = 0.378 - diffs*model_coeffs';
    end
    
    [~,sel] = min(abs(scores));
    sel_ID = subjectId_HUTUBS(sel);

    sel_HRTFs = SOFAload([HUTUBS_path 'subj_' sprintf('%03d', sel_ID) '.sofa']);
    [ITD, delays] = itdestimator(sel_HRTFs);
    pos = sel_HRTFs.SourcePosition(:, 1:2);
    
    if plots
        subplot(211)
        bar(subjectId_HUTUBS, abs(scores))
        subplot(223)
        scatter3(sel_HRTFs.SourcePosition(:,1),sel_HRTFs.SourcePosition(:,2),delays(:,1));
        hold on;
        scatter3(sel_HRTFs.SourcePosition(:,1),sel_HRTFs.SourcePosition(:,2),delays(:,2));
        subplot(224)
        scatter3(sel_HRTFs.SourcePosition(:,1),sel_HRTFs.SourcePosition(:,2),ITD);
    end
end