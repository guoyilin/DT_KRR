
function Node = decisionTree( TrainIndex, stopNum, gamma, lambda, isKmeans, rankNum, percent)
global TrainLabels;
global TrainData;
global globalattriNum;
if(size(TrainIndex, 1) == 0 || size(TrainIndex, 1) == 0) 
     fprintf('\n TrainData is null');
     Node.label = -1;
     return
end
if(size(unique(TrainLabels(TrainIndex,:), 'rows'), 1) == 1)
    [~, Node.label] = max(TrainLabels(TrainIndex(1,:),:));
    Node.name = 'leaf';
    Node.left = NaN;
    Node.right = NaN;
    Node.attri = NaN;
    Node.split = NaN;
    Node.num = size(TrainIndex, 1);
    return
%% use krr.
%elseif(size(TrainData, 1) < stopNum && size(TrainData, 1) >= 1000)%nystrom krr. 
elseif(size(TrainIndex, 1) < stopNum )%nystrom krr.
   [Alpha, train_index, kernel_gamma] = NystromSampling(gamma,TrainIndex, percent*size(TrainIndex,1) , isKmeans, lambda); 
   Representer.alpha = Alpha;
   Representer.data = train_index;
   Representer.gamma = kernel_gamma;
   Node.representer = Representer;
   Node.label = -1;
   Node.left = NaN;
    Node.right = NaN;
    Node.attri = NaN;
    Node.split = NaN;
   Node.name = 'leaf';
    Node.num = size(TrainIndex, 1);
   return
%elseif( size(TrainData, 1) < 1000) % direct krr.
%      [Alpha, train_data, kernel_gamma] = DirectKRR_Representer(TrainData,TrainLabels,gamma,lambda);
%      Representer.alpha = Alpha;
%      Representer.data = train_data;
%      Representer.gamma = kernel_gamma;
%      Node.representer = Representer;
%      Node.label = -1;
%      Node.left = NaN;
%      Node.right = NaN;
%      Node.attri = NaN;
%      Node.split = NaN;
%      Node.name = 'leaf';
%       Node.num = size(TrainLabels, 1);
%      return
end
m = uint32(sqrt(globalattriNum)); % number of split attribute.
%% compute entropy of current node.
current_entropy = entropy(TrainLabels(TrainIndex, :));
%% random select some attributes for splitting.
Attributes = randperm(m);
selectedAttri = Attributes(1, 1:m);
%% find the best split according to information gain.
best_attri = 0;
best_split = 0;
best_ig = -inf;
for i=1:size(selectedAttri,2) % for each attribute
    values = unique(TrainData(TrainIndex, selectedAttri(i)));
    short_values = zeros(1,99);
    for k=1:99
        short_values(k)=k;
    end
    values = prctile(values, short_values);
    values = values';
    %try each possible split value.
    for value  = 1:size(values,1)
        leftIndex = find(TrainData(TrainIndex, selectedAttri(i)) <= values(value, 1));
        leftIndex = TrainIndex(leftIndex,:);
        leftLabels = TrainLabels(leftIndex, :);
        rightIndex = find(TrainData(TrainIndex, selectedAttri(i)) > values(value, 1));
        rightIndex = TrainIndex(rightIndex,:);
        rightLabels = TrainLabels(rightIndex, :);
        left_entropy = entropy(leftLabels);
         right_entropy = entropy(rightLabels);
        ig = current_entropy - size(leftLabels, 1)/size(TrainIndex, 1) * left_entropy - size(rightLabels, 1)/size(TrainIndex,1) * right_entropy;
        if(ig > best_ig)
             best_ig = ig;
            best_split = values(value, 1);
            best_attri = selectedAttri(i);
        end
    end
end
%% information gain is near to zero.
if(best_ig < 1.0e-6)
    [~, Node.label] = max(TrainLabels(TrainIndex(1),:));
    Node.name = 'leaf';
    Node.left = NaN;
    Node.right = NaN;
    Node.attri = NaN;
    Node.split = NaN;
    Node.num = size(TrainIndex, 1);
    return
end
%% binary split.
LeftIndex = find(TrainData(TrainIndex, best_attri) <= best_split);
LeftIndex = TrainIndex(LeftIndex,:);
RightIndex = find(TrainData(TrainIndex, best_attri) > best_split);
RightIndex = TrainIndex(RightIndex,:);
leftNode = decisionTree(LeftIndex,  stopNum, gamma, lambda, isKmeans, rankNum, percent);
rightNode = decisionTree(RightIndex,  stopNum, gamma, lambda, isKmeans, rankNum, percent);
Node.attri = best_attri;
Node.split = best_split;
Node.left = leftNode;
Node.right = rightNode;
Node.name = 'non-leaf';
Node.label = -1;
Node.num = size(TrainIndex, 1);
end

