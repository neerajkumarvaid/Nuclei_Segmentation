require 'dp'
require 'cutorch'
require 'optim'
require 'image'
require 'cunn'
matio = require 'matio'
-- input_image = image.load('/home/sanuj/Projects/nuclei-net/data/testing-data/84_LLM_YR4.jpg', 3, 'byte')
-- s = 51
-- c = (#im)[1]; h = (#im)[2]; w = (#im)[3]

-- out = model:forward(im_tensor):exp()

ws = 51
channels = 3
batch_size = 2000
classes = 3
for n = 1, 4 do
	-- n = 1
	file_name = 'd' .. n
	index = 'a' .. n

	input_mat = matio.load('/home/sanuj/Projects/nuclei-net/data/testing-data/d/'.. file_name ..'_02.mat')
	num_images = (#input_mat[index])[1]

	image_tensor = torch.Tensor(num_images, channels, ws, ws)
	for i = 1, num_images do
		image_tensor[{i, {}, {}, {}}] = torch.reshape(input_mat[index][i], 1, channels, ws, ws)[1]
	end

	xp = torch.load('/home/sanuj/save/amitpc-HP-Z620-Workstation_1454011145_1.dat')
	model = xp:model()

	labels = torch.Tensor(num_images, classes)
	for i = 1, num_images, batch_size do
		temp = model:forward(image_tensor[{ {i, i+batch_size-1}, {}, {}, {} }]):exp()
		labels[{ {i, i+batch_size-1}, {} }] = temp:double()
	end

	for i = 1, channels do
		matio.save('/home/sanuj/Projects/nuclei-net/data/testing-data/d/' .. file_name .. '_02_' .. i .. '.mat', labels[{ {}, {i} }])
	end
end