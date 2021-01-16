function new_delays = interpolateDelays(delays, old_pos, new_pos)
    % allocate variable
    new_delays = NaN(size(new_pos, 1), 2);
    % ignore radius
    new_pos = new_pos(:,1:2);
    % for each new elevation...
    elevations = unique(new_pos(:,2));
    for el = elevations'
        % old positions for current elevation
        old_el_idx = find(old_pos(:,2) == el);
        % new positions for current elevation
        new_el_idx = find(new_pos(:,2) == el);
        % create arguments for interp1q
        xi = new_pos(new_el_idx,1);
        x = old_pos(old_el_idx,1);
        y = delays(old_el_idx,:);
        % duplicate first point (needed to interpolate last azimuths)
        x = [x; 360+x(1)];
        y = [y; y(1,:)];
        % interpolate and store
        new_delays(new_el_idx, :) = interp1q(x, y, xi);  
    end
end