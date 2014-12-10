function result = classify(TestIndex, tree)
global TestData;
global TestLabels;
if(tree.label ~=-1 && strcmp(tree.name,'leaf'))
    [~, I2] = max(TestLabels(TestIndex,:), [], 2);
    result = size(find(I2==tree.label), 1);
    return
elseif(strcmp(tree.name,'leaf') && isfield(tree, 'representer'))
     Values = DirectKRRPredict(tree.representer.alpha, tree.representer.data, tree.representer.gamma, TestIndex);
     [~, I1] = max(Values,[], 2);
     %disp(TestIndex');% TestIndex is wrong.
     [~, I2] = max(TestLabels(TestIndex,:), [], 2);
     index = I1 == I2;
     result = size(find(index == 1), 1);
     return
end
LeftIndex = find(TestData(TestIndex, tree.attri(1,1)) <= tree.split(1,1));
LeftIndex = TestIndex(LeftIndex,:);
l1 = classify(LeftIndex, tree.left);

RightIndex = find(TestData(TestIndex, tree.attri(1,1)) > tree.split(1,1));
RightIndex = TestIndex(RightIndex,:);
l2 = classify(RightIndex, tree.right);
result = l1 + l2;
return
end

