require 'dp'
require 'cutorch'
require 'optim'
require 'image'
require 'cunn'
require 'os'

matio = require 'matio'

ws = 51
batch_size = 2000
classes = 3
image_dir = '/data/testing-data/40x'
image_name = 'PrognosisTMABlock1_A_3_1_H&E.png'
-- image_dir = '/home/sanuj/Projects/nuclei-net/data/testing-data'
-- image_name = '84_LLM_YR4_002.png'

input_image = image.load(image_dir .. '/' .. image_name, 3, 'byte')
channels = (#input_image)[1]; w = (#input_image)[2]; h = (#input_image)[3]

-- file_num = 1
-- p = (ws-1)/2
os.execute("mkdir " .. image_dir .. '/' .. 'tmp')
xp = torch.load('/home/sanuj/save/amitpc-HP-Z620-Workstation:1455425081:2.dat')
-- xp = torch.load('/home/sanuj/save/LYoga:1454304060:1.dat')
model = xp:model()

-- print((h-ws+1)*(w-ws+1)*channels*ws*ws)
cropped = torch.Tensor((h-ws+1)*(w-ws+1), channels, ws, ws):byte()
labels = torch.Tensor((h-ws+1)*(w-ws+1), classes)
-- batch_done = true
counter = 1
last_counter = 1
for x = 0, h-ws do
	for y = 0, w-ws do
		-- if temp ~= nil then
		-- 	print('Cropping at x: ' .. x .. ', y: ' .. y .. ', len: ' .. (#temp)[1])
		-- end
		print('Counter: ' .. counter)
		cropped[{ {counter}, {}, {}, {} }] = image.crop(input_image, x, y, x+ws, y+ws)
		-- cropped = torch.reshape(cropped, 1, channels, ws, ws)
		-- if batch_done then
		-- 	temp = cropped
		-- 	batch_done = false
		-- else
		-- 	temp = torch.cat(temp, cropped, 1)
		-- end
		-- if (#temp)[1] == batch_size then
		-- 	-- print('SAVING File Number: ' .. file_num)
		-- 	-- matio.save(image_dir .. '/tmp/' .. file_num .. '.mat', temp)
		-- 	-- file_num = file_num + 1
		-- 	print('PREDICTING!!')
		-- 	temp_labels = model:forward(temp):exp()
		-- 	if labels == nil then
		-- 		labels = temp_labels
		-- 	else
		-- 		labels = torch.cat(labels, temp_labels)
		-- 	end
		-- 	-- temp = nil
		-- 	batch_done = true
		-- end
		if counter % batch_size == 0 then
			print('PREDICTING!!!')
			temp = model:forward(cropped[{ {counter-batch_size+1, counter}, {}, {}, {} }]):exp()
			labels[{ {counter-batch_size+1, counter}, {} }] = temp:double()
			last_counter = counter
		end
		counter = counter + 1
	end
end

if last_counter ~= (counter - 1) then
	temp = model:forward(cropped[{ {last_counter+1, counter-1}, {}, {}, {} }]):exp()
	labels[{ {last_counter+1, counter-1}, {} }] = temp:double()
end
-- if temp ~= nil then
-- 	print('PREDICTING!!')
--     temp_labels = model:forward(temp):exp()
-- 	if labels == nil then
-- 		labels = temp_labels
-- 	else
-- 		labels = torch.cat(labels, temp_labels)
-- 	end
-- 	temp = nil
-- end

for i = 1, channels do
	image.save(image_dir .. '/tmp/' .. i .. '.png', image.vflip(torch.reshape(labels[{ {}, {i} }], h-ws+1, w-ws+1)))
	-- image.save(image_dir .. '/tmp/' .. i .. '.png', image.hflip(image.rotate(torch.reshape(labels[{ {}, {i} }], h-ws+1, w-ws+1), 3*math.pi/2)))
	-- matio.save(image_dir .. '/tmp/' .. i .. '.mat', labels[{ {}, {i} }])
end

-- for n = 1, 4 do
--     -- n = 1
--     file_name = 'd' .. n
--     index = 'a' .. n

--     input_mat = matio.load('/home/sanuj/Projects/nuclei-net/data/testing-data/d/'.. file_name ..'_02.mat')
--     num_images = (#input_mat[index])[1]

--     image_tensor = torch.Tensor(num_images, channels, ws, ws)
--     for i = 1, num_images do
--         image_tensor[{i, {}, {}, {}}] = torch.reshape(input_mat[index][i], 1, channels, ws, ws)[1]
--     end

--     xp = torch.load('/home/sanuj/save/amitpc-HP-Z620-Workstation_1454011145_1.dat')
--     model = xp:model()

--     labels = torch.Tensor(num_images, classes)
--     for i = 1, num_images, batch_size do
--         if i+batch_size-1 <= num_images then
--             temp = model:forward(image_tensor[{ {i, i+batch_size-1}, {}, {}, {} }]):exp()
--             labels[{ {i, i+batch_size-1}, {} }] = temp:double()
--         else
--             temp = model:forward(image_tensor[{ {i, num_images}, {}, {}, {} }]):exp()
--             labels[{ {i, num_images}, {} }] = temp:double()
--     end

--     for i = 1, channels do
--         matio.save('/home/sanuj/Projects/nuclei-net/data/testing-data/d/' .. file_name .. '_02_' .. i .. '.mat', labels[{ {}, {i} }])
--     end
-- end
