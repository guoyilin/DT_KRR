function [idx, center, m] = eff_kmeans(data, m, MaxIter);

[n, dim] = size(data);
dex = randperm(n);
center = data(dex(1:m),:);
threshold = inf;
previous_center = center;
for i = 1:MaxIter;
    nul = zeros(m,1);
    [xx, idx] = min(sqdist(center', data'));
    for j = 1:m;
        dex = find(idx == j);% data belongs to cluster j
        l = length(dex); 
        cltr = data(dex,:);
        if l > 1;
            center(j,:) = mean(cltr);% update cluster j's center
        elseif l == 1;
            center(j,:) = cltr;
        else
            nul(j) = 1;%cluster j is empty
        end;
    end;
    dex = find(nul == 0);
    m = length(dex);
    %check the threshold change
    threshold=sqrt(sum(abs(center - previous_center).^2, 2));
    threshold = sum(threshold);
    if(threshold < 0.1)
        center = center(dex,:);
        return
    else
       center = center(dex,:); 
       previous_center = center;
    end
    
end;

