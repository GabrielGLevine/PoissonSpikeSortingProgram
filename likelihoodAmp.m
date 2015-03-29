function [score,ISIScore,AmpScore]=likelihoodAmp(trains,neurons)
[neuronNum,maxTime]=size(trains);
isiScore=zeros(1,neuronNum);
ampScore=zeros(1,neuronNum);
trainScore=zeros(1,neuronNum);
parfor i=1:neuronNum
    spikes=find(trains(i,:));
    for j=1:length(spikes)-1
        isiScore(i)=isiScore(i)+log(poisspdf(spikes(j+1)-spikes(j),neurons.isi(i)));
        ampScore(i)=ampScore(i)+log(normpdf(trains(spikes(j)),neurons.Amplitudes(i),0.4));
    end
    trainScore(i)=40*log(normpdf(length(spikes),max(neurons.spikes(1,:))/neurons.isi(i),(max(neurons.spikes(1,:))/neurons.isi(i))/8));
end
score=-(sum(isiScore)+sum(ampScore)+sum(trainScore));
AmpScore=-sum(ampScore);
ISIScore=-sum(isiScore);
end