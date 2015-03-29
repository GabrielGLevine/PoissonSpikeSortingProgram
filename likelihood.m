function score=likelihood(trains,neurons)
[neuronNum,maxTime]=size(trains);
isiScore=zeros(1,3);
parfor i=1:neuronNum
    spikes=find(trains(i,:));
    for j=1:length(spikes)-1
%        isiScore(i)=isiScore(i)+log(poisspdf(spikes(j+1)-spikes(j),neurons.isi(i)))+log(normpdf(trains(spikes(j)),neurons.Amplitudes(i),0.3));
         isiScore(i)=isiScore(i)+log(poisspdf(spikes(j+1)-spikes(j),neurons.isi(i)));
    end
end
score=-sum(isiScore);
end