function initTrains=initialSpikes(spikes,neuronNum)
initTrains=zeros(neuronNum,length(spikes));
for i=1:length(spikes)
    if spikes(i)~=0
        for j=1:spikes(i)
            r1=randi([1 neuronNum]);
            initTrains(r1,i)=initTrains(r1,i)+1;
        end
    end
end