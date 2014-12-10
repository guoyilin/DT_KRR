function current_entropy = entropy( trainLabels )
%compute entropy
pro_labels = sum(trainLabels==1, 1)./ size(trainLabels,1);
if(size(pro_labels, 2) < 10)
    disp('hello');
end
pro_labels = pro_labels(find(pro_labels~=0));
current_entropy = -sum(pro_labels.*log2(pro_labels));
end

