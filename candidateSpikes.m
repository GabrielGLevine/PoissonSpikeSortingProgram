function candTrains=candidateSpikes(trains,inds,neuronNum,swapCount)
candTrains=trains;
for j=1:3
    r1=randi([1 length(inds)]);
    spikeInds=find(candTrains(:,inds(r1)));
    r2=randi([1 length(spikeInds)]);
    r4=randi([1 candTrains(spikeInds(r2),inds(r1))]);
    candTrains(spikeInds(r2),inds(r1))=candTrains(spikeInds(r2),inds(r1))-r4;
    r3=randi([1 neuronNum]);
    candTrains(r3,inds(r1))=candTrains(r3,inds(r1))+r4;
end
end

% candTrains=trains;
% for j=1:swapCount
%     r1=randi([1 length(inds)]);
%     spikeInds=find(candTrains(:,inds(r1))~=0);
%     r2=randi([1 length(spikeInds)]);
%     candTrains(spikeInds(r2),inds(r1))=candTrains(spikeInds(r2),inds(r1))-1;
%     r3=randi([1 neuronNum]);
%     candTrains(r3,inds(r1))=candTrains(r3,inds(r1))+1;
% end