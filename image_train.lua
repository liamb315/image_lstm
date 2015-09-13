
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local LSTM = require 'LSTM' 
local model_utils = require 'model_utils'
local ImageLoader = require 'ImageLoader'
local CreateTensors = require 'CreateTensors'

-- Options of the model
local opt = {print_every     = 1,
 			 x_csv           = "x_rep", --Just use 'filename' for 'filename.csv'
             y_csv           = "y_rep", 
  			 seed            = 1,
  			 batch_size      = 16,
  			 rnn_size        = 10,
  			 output_size     = 10,
  			 savefile        = "snapshots/model_snapshot.t7",
  			 save_every      = 100,
  			 seq_length      = 39,
  			 max_epochs      = 10,
             init_learn_rate = 1,  
             dec_learn_rate  = 1000,
             dec_rate_by     = 0.5
             }

print(opt)

torch.manualSeed(opt.seed)

-- Generate tensors
CreateTensors.generateTensors(opt.x_csv)
CreateTensors.generateTensors(opt.y_csv)

-- load data from tensors
local x_tensor = 'tensors/'..opt.x_csv..'.th7'
local y_tensor = 'tensors/'..opt.y_csv..'.th7'

local loader = ImageLoader.create(x_tensor, y_tensor, opt.batch_size, opt.seq_length, opt.rnn_size)

-- define model prototypes for ONE timestep, then clone them
local protos = {}
-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
protos.lstm = LSTM.lstm(opt)
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, opt.output_size)):add(nn.LogSoftMax())
protos.criterion = nn.ClassNLLCriterion()

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.softmax)
params:uniform(-0.08, 0.08)

-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
local initstate_h = initstate_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()



-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    --TODO: Retrieve a batch from the data
    local x, y = loader:next_image_batch() 

    --print('x')
    --print(x)
    --print('y')
    --print(y)

    ------------------- forward pass -------------------
    local example = {}   
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,opt.seq_length do
        example[t] = x[{{}, t}]

        --print('example['..t..']')
        --print(example[t])

    	lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{example[t], lstm_c[t-1], lstm_h[t-1]})
    	predictions[t] = clones.softmax[t]:forward(lstm_h[t])
        --print('predictions['..t..']')
        --print(predictions[t])
        _, i = torch.max(predictions[t], 2)
        --print('Actual and Predicted')
        --print(y[{{}, t}])
        --print(lstm_c[t], lstm_h[t])
        --print(clones.criterion[t]:forward(predictions[t], y[{{}, t}]))  
        --print(y[{{}, t}])
        
        -- OLD Loss
        -- Add the average loss across the batch
        --loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])

        -- NEW Loss based only on the last example of sequence
        if t == opt.seq_length then
            loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
        end
        --print('loss: '.. loss)
    end

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        --local doutput_t = clones.criterion[t]:backward(predictions[t], y[2][t])
        -- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the softmax (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the softmax and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == opt.seq_length then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
        else
            dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))
        end

        -- TODO
        -- We don't want to do backpropagation through embeddings now!
        -- backprop through LSTM timestep
        _, dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {example[t], lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))
    end


    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end


-- optimization stuff
local losses = {}
local optim_state = {learningRate = opt.init_learn_rate}
--TODO: What is loader.nbatches, why not retrieve from opt?
--local iterations =  10
local iterations = opt.max_epochs * loader.nbatches 

for i = 1, iterations do
    --print('length of params')
    --print(#params)
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
    end

    
    if i % opt.dec_learn_rate == 0 then
        optim_state.learningRate = optim_state.learningRate*opt.dec_rate_by
    end    
    
    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
    end
end




