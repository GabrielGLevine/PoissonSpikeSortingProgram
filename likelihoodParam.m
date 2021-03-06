function score=likelihoodParam(trains,params)
[neuronNum,maxTime]=size(trains);
isiScore=zeros(1,3);
for i=1:neuronNum
    spikes=find(trains(i,:));
    for j=1:length(spikes)-1
%        isiScore(i)=isiScore(i)+log(poisspdf(spikes(j+1)-spikes(j),neurons.isi(i)))+log(normpdf(trains(spikes(j)),neurons.Amplitudes(i),0.3));
         isiScore(i)=isiScore(i)+log(poisspdf(spikes(j+1)-spikes(j),params(i)));
    end
end
score=-sum(isiScore);
end






% function [score,mean]=likelihoodParam(trains,neurons)
% paramScore=zeros(1,2);
% mean=zeros(1,2);
% parfor i=1:2
%     [mean(i), conInt]=poissfit(diff(find(trains(i,:))));
%     paramScore(i)=(conInt(2)-conInt(1))/mean(i);
% end
% score=sum(paramScore);