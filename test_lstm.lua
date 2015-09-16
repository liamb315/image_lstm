require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

local CreateTensors = require 'CreateTensors'
local ImageLoader   = require 'ImageLoader'

local opt = {seed        = 1,
			 csv_path    = 'data/',
			 tensor_path = 'tensors/',
			 output_path = 'output/',
			 model       = 'snapshots/model_snapshot.t7',
			 x_csv       = 'x_test_mini',  --Just use 'filename' for 'filename.csv'
			 y_csv       = 'y_test_mini',  --Just use 'filename' for 'filename.csv'
			 seq_length  = 39,
			 batch_size  = 1,              -- TOOD: Check this
			 input_size  = 4096,
			 rnn_size    = 256
			}

print(opt)
torch.manualSeed(opt.seed)

-- Generate tensors (needed if the tensors are not already created)
--CreateTensors.generateTensors(opt.x_csv, opt.csv_path, opt.tensor_path)
--CreateTensors.generateTensors(opt.y_csv, opt.csv_path, opt.tensor_path)

-- load data from tensors
local x_tensor = opt.tensor_path .. opt.x_csv .. '.th7'
local y_tensor = opt.tensor_path .. opt.y_csv .. '.th7'

-- Load the model
local protos = torch.load(opt.model)

-- TODO:  
--  Create loader for test data.
--    * Store 3-Tensor
--    * We then will simply move across 2-Tensors 
local loader = ImageLoader.create(x_tensor, y_tensor, opt.batch_size, opt.seq_length, opt.input_size)


------------------- Predictions -------------------
-- TODO
--  Retrieve seq_length from the 2-Tensor
--local seq_length = x_tensor:size(1)
local seq_length = opt.seq_length

-- LSTM initial state, note that we're using minibatches OF SIZE ONE here
local prev_c = torch.zeros(1, opt.rnn_size)
local prev_h = prev_c:clone()


function evaluate_batch()
	local batch_correct = 0

	------------------ get minibatch -------------------
	local x, y = loader:next_image_batch() 
	--print(x)
	

	------------------ predict batch -------------------
	for t = 1, seq_length do		
		--print(x[{{}, t}])
		local linear_net     = protos.linear:forward(x[{{}, t}])
		local next_c, next_h = unpack(protos.lstm:forward{linear_net, prev_c, prev_h})
		prev_c:copy(next_c)
	    prev_h:copy(next_h)
		
	    if t == seq_length then
			local log_probs = protos.softmax:forward(next_h)
			local _, argmax = torch.max(log_probs, 2)
			
			--print('Actual    :', y[{{}, t}][1])
			--print('Prediction:', argmax[1][1])	
			print(y[{{}, t}][1], argmax[1][1])

			if y[{{}, t}][1] == argmax[1][1] then
				batch_correct = batch_correct + 1
			end
			--print('softmax weights')
			--print(protos.softmax:parameters()[1])
		end
	end
	return batch_correct
end

print('Total properties correct: ', correct)

local total_correct = 0
local iterations    = loader.nbatches

for i = 1, iterations do
	total_correct = total_correct + evaluate_batch()

end

-- TODO:  Fix this
print('Percent correct ', total_correct/(iterations/opt.batch_size))



