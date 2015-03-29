function [score,isiScore,ampScore]=objfunc(trains,spikesPer,neuronNum,neurons)
isiScore=0;
ampScore=0;
for i=1:neuronNum
    isiScore=isiScore+sqrt((neurons.isi(i)-poissfit(diff(find(trains(i,:)))))^2);
    %ampScore=ampScore+sqrt((spikesPer-sum(trains(i,:)))^2);
end
score=isiScore;
end