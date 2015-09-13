require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local CreateTensors = require 'CreateTensors'
local ImageLoader = require 'ImageLoader'

local opt = {seed       = 123,
			 model      = 'snapshots/model_snapshot.t7',
			 x_csv      = 'x_mini',  --Just use 'filename' for 'filename.csv'
			 y_csv      = 'y_mini'   --Just use 'filename' for 'filename.csv'
             }

print(opt)

-- preparation and loading
torch.manualSeed(opt.seed)

-- Read in the files for testing
CreateTensors.generateTensors(opt.x_csv)
CreateTensors.generateTensors(opt.y_csv)

-- load data from tensors
local x_tensor = torch.load('tensors/'..opt.x_csv..'.th7')
local y_tensor = torch.load('tensors/'..opt.y_csv..'.th7')

-- Load the model
protos = torch.load(opt.model)


-- TODO:  We want a different type of loader, we don't want to load fixed length sequences!
-- local loader = ImageLoader.create(x_tensor, y_tensor, opt.batch_size, opt.seq_length, opt.rnn_size)


----------------------
-- Core functionality
----------------------
local opt = {rnn_size   = 10,
			 seq_length = x_tensor:size(1)
			}

-- LSTM initial state, note that we're using minibatches OF SIZE ONE here
local prev_c = torch.zeros(1, opt.rnn_size)
local prev_h = prev_c:clone()

for t = 1, opt.seq_length do
	print('Iteration: '.. t)
	--print()
	--print('prev_c, prev_h', prev_c, prev_h)
	local next_c, next_h = unpack(protos.lstm:forward{x_tensor[t], prev_c, prev_h})
	prev_c:copy(next_c)
    prev_h:copy(next_h)
	
    --print('lstm weights')
    --print(protos.lstm:parameters())

	--print('next_c, next_h', next_c, next_h)
	local log_probs = protos.softmax:forward(next_h)
	--print('softmax weights')
	--print(protos.softmax:parameters()[1])
	--print('log-probs')
	--print(log_probs)
	local _, i = log_probs:max(1)
	print('prediction:')
	print(i)
	print('actual:')
	print(y_tensor[t])
end


-- seq_length will be now given by the number of consecutive images for a single property
-- TODO:  seq_length = 

--[[
for t = 1, seq_length do
	example[t] = x[{{}, t}]
	lstm_c[t], lstm_h[t] = unpack(protos.lstm[t]:forward{example[t], lstm_c[t-1], lstm_h[t-1]})
	predictions[t] = protos.softmax[t]:forward(lstm_h[t])
	_, i = torch.max(predictions[t], 2)
end
--]]


-----------------------------------
-- Write the outputs to text files 
-----------------------------------


