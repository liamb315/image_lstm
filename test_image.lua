require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

local CreateTensors = require 'CreateTensors'
local ImageLoader = require 'ImageLoader'

local opt = {seed       = 1,
			 model      = 'snapshots/model_snapshot.t7',
			 x_csv      = 'x_mini',  --Just use 'filename' for 'filename.csv'
			 y_csv      = 'y_mini',  --Just use 'filename' for 'filename.csv'
			 rnn_size   = 256
			}

print(opt)
torch.manualSeed(opt.seed)

-- Read in the files for testing
CreateTensors.generateTensors(opt.x_csv)
CreateTensors.generateTensors(opt.y_csv)

-- load data from tensors
local x_tensor = torch.load('tensors/'..opt.x_csv..'.th7')
local y_tensor = torch.load('tensors/'..opt.y_csv..'.th7')

-- Load the model
protos = torch.load(opt.model)

-- TODO:  
--  Create loader for test data.
--    * Store 3-Tensor
--    * We then will simply move across 2-Tensors 
-- local loader = ImageLoader.create(x_tensor, y_tensor, opt.batch_size, opt.seq_length, opt.rnn_size)

------------------- Predictions -------------------
-- TODO
--  Retrieve seq_length from the 2-Tensor
local seq_length = x_tensor:size(1)

-- LSTM initial state, note that we're using minibatches OF SIZE ONE here
local prev_c = torch.zeros(1, opt.rnn_size)
local prev_h = prev_c:clone()

for t = 1, seq_length do
	print('Iteration: '.. t)
	-- TOOD
	--  Insert the linear layer (saved in protos)
	--  Pass this to the LSTM 
	local next_c, next_h = unpack(protos.lstm:forward{x_tensor[t], prev_c, prev_h})
	prev_c:copy(next_c)
    prev_h:copy(next_h)
	
    --print('lstm weights')
    --print(protos.lstm:parameters())

    if t == seq_length then
		local log_probs = protos.softmax:forward(next_h)
		local _, argmax = log_probs:max(1)
		print('Actual:', y_tensor[t])
		print('Prediction:', argmax)	
		--print('softmax weights')
		--print(protos.softmax:parameters()[1])
		--print('log-probs')
		--print(log_probs)
	end
end


------------------- Output -------------------


