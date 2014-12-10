function[Alpha, train_index, kernel_gamma]  = NystromSampling( gamma,trainIndex, m, s, lambda )
global TrainData;
global TrainLabels;
%Input:
% data: n-by-dim data matrix;
% m: number of landmark points;
% s: 'r' for random sampling and 'k' for k-means based sampling
%gamma = stdv(data);
%disp(['gamma:', num2str(gamma)]);
%lambda = 1 / (gamma * size(data, 1)); it's not good.
%disp(['lambda:', num2str(lambda)]);
n = size(trainIndex, 1);
m = uint16(m);
if(s == 'k')
    [idx, center, m] = eff_kmeans(TrainData(trainIndex,:), m, 10); %#iteration is restricted to 5
end

if(s == 'r')
   dex = randperm(n);
   center = TrainData(trainIndex(dex(1:m), 1),:);
end

%if(kernel.type == 'pol');
%    W = center * center';
%    E = data * center';
%    W = W.^kernel.para;
%    E = E.^kernel.para;
%end;

%if(kernel.type == 'rbf');
    W = exp(- sqdist(center', center')/gamma);
    E = exp(- sqdist(TrainData(trainIndex, :)', center')/gamma);
%end;

[Ve, Va] = eig(W);
va = diag(Va); % eigen values.
%size1 = size(find(va > 1e-6), 1);
%[~, sortIndex] = sort(va(:), 'descend');
%selected_count = 100;
%if(size1 < selected_count)
%    selected_count = size1;
%end
%pidx = sortIndex(1:selected_count);
pidx = find(va > 1e-6);
%inVa = sparse(diag(va(pidx).^(-0.5))); % inverse
inVa = diag(va(pidx).^(-0.5)); % inverse
G = E * Ve(:,pidx) * inVa;
Ktilde = G * G';
%full_K = exp(-sqdist(data', data')/gamma);
%err1 = norm(full_K - Ktilde, 'fro');
%disp(['error:', num2str(err1)]);
Alpha = inv(Ktilde + lambda*eye(size(Ktilde)))*TrainLabels(trainIndex,:); % N * T, out of memory.
train_index = trainIndex;
kernel_gamma = gamma;
end

