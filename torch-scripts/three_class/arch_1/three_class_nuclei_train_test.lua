require 'torch';
require 'nn';
require 'itorch';
require 'image'

trainset = torch.load('/home/sanuj/Projects/nuclei-net/data/training-data/78_RLM_YR4_3_class/train_small.t7')
testset = torch.load('/home/sanuj/Projects/nuclei-net/data/training-data/63_LLM_YR4_3_class/test_small.t7')

--image.display(trainset[1][100])
--print(trainset[2][100])

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t[1][i], t[2][i]} 
                end}
);

function trainset:size() 
    return self.data:size(1) 
end

trainset[1] = trainset[1]:double()
testset[1] = testset[1]:double()

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 48, 6, 6)) -- 3 input image channel, 48 output channels, 6x6 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(48, 48, 4, 4))  -- 48 input image channel, 48 output channels, 4x4 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(48*10*10))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(48*10*10, 1024))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(1024, 1024))
net:add(nn.Linear(1024, 3))                   -- 3 is the number of outputs of the network
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

print('Nuclei-net\n' .. net:__tostring());

criterion = nn.ClassNLLCriterion()

--require 'cunn'
--net = net:cuda()
--criterion = criterion:cuda()
--trainset[1] = trainset[1]:cuda()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10 -- just do 10 epochs of training.

class_performance = {0, 0, 0}
for i=1,30000 do
    local groundtruth = testset[2][i]
    local prediction = net:forward(testset[1][i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,3 do
    print(i, 100*class_performance[i]/10000 .. ' %')
end
