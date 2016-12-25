require 'dp'
require 'cutorch'
require 'optim'
require 'image'
require 'cunn'
require 'os'

-- matio = require 'matio'

ws = 51
batch_size = 1500 --1600
classes = 3

-- image_dir = '/home/sanuj/Downloads/anmol_maps'
-- image_name = 'K10_13332_8866.jpg'

image_dir = '/data/testing-data/40x'
image_name = '63_LLM_YR4_cropped.jpg'

input_image = image.load(image_dir .. '/' .. image_name, 3, 'byte')
channels = (#input_image)[1]; w = (#input_image)[2]; h = (#input_image)[3]

-- mirror pad
-- p = ws-1
-- im = torch.ByteTensor(channels, w+p, h+p):zero()
-- im[{ {}, {p/2+1, w+p/2}, {p/2+1, h+p/2} }] = input_image
-- h = h+p
-- w = w+p
------------------------------------------------------------
p = ws-1
module = nn.SpatialReflectionPadding(p/2, p/2, p/2, p/2)
module:cuda()
im = module:forward(input_image:cuda())
im = im:byte()
h = h+p
w = w+p
------------------------------------------------------------

-- map_path = '/home/sanuj/Downloads/anmol_maps/K10N_13332_8866_binary.jpg'
-- map = image.load(map_path)
-- map = map:byte()

-- for x = 1, (#map)[2] do
-- 	for y = 1, (#map)[3] do
-- 		print('x: ' .. x .. 'y: ' .. y)
-- 		if map[1][x][y] == 0 then
-- 			im[1][x][y] = 1
-- 			im[2][x][y] = 1
-- 			im[3][x][y] = 1
-- 		end
-- 	end
-- end

-- print('Done superimposing.')

-- file_num = 1
-- p = (ws-1)/2
os.execute("mkdir " .. image_dir .. '/' .. 'results')
-- xp = torch.load('/home/sanuj/save/amitpc-HP-Z620-Workstation:1455425081:2.dat')
-- xp = torch.load('/home/sanuj/Projects/models/amitpc-HP-Z620-Workstation:1457692046:1.dat') -- final 20x
-- /home/sanuj/Projects/models/train_7259.dat
xp = torch.load('/home/sanuj/Projects/models/train_701_val_734.dat') -- latest 20x
-- xp = torch.load('/home/sanuj/save/LYoga:1462988633:1.dat') -- new 20x
-- xp = torch.load('/home/sanuj/save/amitpc-HP-Z620-Workstation_1454011145_1.dat') -- final 40x
-- xp = torch.load('/home/sanuj/save/LYoga:1454304060:1.dat')
model = xp:model()

-- print((h-ws+1)*(w-ws+1)*channels*ws*ws)
-- cropped = torch.Tensor((h-ws+1)*(w-ws+1), channels, ws, ws):byte()
-- labels = torch.Tensor((h-ws+1)*(w-ws+1), classes)

cropped = torch.Tensor(batch_size, channels, ws, ws):byte()
labels = torch.Tensor((h-ws+1)*(w-ws+1), classes)

-- batch_done = true
counter = 0
last_counter = 1

for x = 0, h-ws do
	for y = 0, w-ws do
		print('Counter: ' .. counter .. ' cropped: ' .. (counter % batch_size)+1)
		cropped[{ {(counter % batch_size)+1}, {}, {}, {} }] = image.crop(im, x, y, x+ws, y+ws)
		if (counter+1) % batch_size == 0 then
			print('PREDICTING!!!')
			temp = model:forward(cropped[{ {1, batch_size}, {}, {}, {} }]):exp()
			labels[{ {(counter+1)-batch_size+1, counter+1}, {} }] = temp:double()
			last_counter = counter
		end
		counter = counter + 1
	end
end

if last_counter ~= (counter - 1) then
	temp = model:forward(cropped[{ {1, counter % batch_size}, {}, {}, {} }]):exp()
	labels[{ {last_counter+2, counter}, {} }] = temp:double()
end

for i = 1, channels do
	image.save(image_dir .. '/results/' .. i .. '.png', image.vflip(torch.reshape(labels[{ {}, {i} }], h-ws+1, w-ws+1)))
end
