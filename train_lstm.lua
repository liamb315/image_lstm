
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

local LSTM = require 'LSTM' 
local model_utils = require 'model_utils'
local ImageLoader = require 'ImageLoader'
local CreateTensors = require 'CreateTensors'

-- Options of the model
local opt = {print_every      = 1,
             csv_path         = "data/",
             tensor_path      = "tensors/",
             x_csv            = "x_train_selldecile_mini", --Just use 'filename' for 'filename.csv'
             y_csv            = "y_train_selldecile_mini", 
             seed             = 1,
             batch_size       = 16,
             input_size       = 4096,
             rnn_size         = 256,
             output_size      = 10,
             savefile         = "snapshots/model_snapshot.t7",
             save_every       = 100,
             seq_length       = 39,
             max_epochs       = 500,
             init_learn_rate  = 1E-1,
             learn_rate_decay = 1E-4  --https://github.com/torch/optim/blob/master/adagrad.lua
             }

print(opt)
torch.manualSeed(opt.seed)

-- Generate tensors (needed if the tensors are not already created)
CreateTensors.generateTensors(opt.x_csv, opt.csv_path, opt.tensor_path)
CreateTensors.generateTensors(opt.y_csv, opt.csv_path, opt.tensor_path)

-- Load data from tensors
local x_tensor = opt.tensor_path .. opt.x_csv .. '.th7'
local y_tensor = opt.tensor_path .. opt.y_csv .. '.th7'

local loader = ImageLoader.create(x_tensor, y_tensor, opt.batch_size, opt.seq_length, opt.input_size)

-- Define model prototypes for ONE timestep, then clone them
local protos     = {}
protos.linear    = nn.Linear(opt.input_size, opt.rnn_size)
protos.lstm      = LSTM.lstm(opt)  --lstm input: {x, prev_c, prev_h}, output: {next_c, next_h}
protos.softmax   = nn.Sequential():add(nn.Linear(opt.rnn_size, opt.output_size)):add(nn.LogSoftMax())
protos.criterion = nn.ClassNLLCriterion()

-- Put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.softmax)
params:uniform(-0.08, 0.08)

-- Make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c   = torch.zeros(opt.batch_size, opt.rnn_size)
local initstate_h   = initstate_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()


-- fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_image_batch() 

    ------------------- forward pass -------------------
    local linear_nets = {}
    local lstm_c      = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h      = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}                -- softmax outputs
    local loss        = 0

    for t=1,opt.seq_length do
        -- Pass data to a linear forward neural network
        linear_nets[t] = clones.linear[t]:forward(x[{{}, t}])
    	lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{linear_nets[t], lstm_c[t-1], lstm_h[t-1]})
    	predictions[t] = clones.softmax[t]:forward(lstm_h[t])
        
        _, argmax = torch.max(predictions[t], 2)
        --print('Actual', y[{{}, t}])
        --print('Predicted', argmax)
        --print(lstm_c[t], lstm_h[t])
        
        -- Loss based only on the last example of sequence
        if t == opt.seq_length then
            loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
        end
    end


    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local rev_linear_nets = {}
    local dlstm_c         = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h         = {}                                  -- output values of LSTM

    for t=opt.seq_length,1,-1 do
        -- Backprop through loss and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])

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

        -- Backprop through LSTM timestep
        rev_linear_nets[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {linear_nets[t], lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))

        -- Backprop through the initial linear layer
        clones.linear[t]:backward(x[{{}, t}], rev_linear_nets[t])
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
local optim_state = {learningRate = opt.init_learn_rate, learningRateDecay = opt.learn_rate_decay}
local iterations = opt.max_epochs * loader.nbatches 

for i = 1, iterations do
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
    end

    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
    end
end




