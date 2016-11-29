
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local model_utils = require 'util.model_utils'

local SessionDataLoader = require 'SessionDataLoader'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
cmd:argument('-test_file_path','path to the test file')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

function get_target_scores(scores, y)
	assert (scores:size()[1] == y:size()[1])
	assert (scores:size()[1] == checkpoint.opt.batch_size)
	
	target_scores = torch.zeros(checkpoint.opt.batch_size, 1)
	for i = 1, checkpoint.opt.batch_size do
		target_scores[i] = scores[i][y[i]]
	end
	
	return target_scores
end
-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
  if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then gprint('package cunn not found!') end
  if not ok2 then gprint('package cutorch not found!') end
  if ok and ok2 then
    gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
    gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  else
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
  local ok, cunn = pcall(require, 'clnn')
  local ok2, cutorch = pcall(require, 'cltorch')
  if not ok then print('package clnn not found!') end
  if not ok2 then print('package cltorch not found!') end
  if ok and ok2 then
    gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
    gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
    cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    torch.manualSeed(opt.seed)
  else
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
  gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn_encoder:evaluate() -- put in eval mode so that dropout works properly
protos.rnn_decoder:evaluate()
protos.attention:evaluate()
protos.rnn_projection:evaluate() -- put in eval mode so that dropout works properly

local attentions = model_utils.clone_many_times(protos.attention, 300, not protos.attention.parameters)

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local encoder_init_state
encoder_init_state = {}
for L = 1,checkpoint.opt.num_layers do
  -- c and h for all layers
  local h_init = torch.zeros(checkpoint.opt.batch_size, checkpoint.opt.rnn_size):double()
  if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
  if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
  table.insert(encoder_init_state, h_init:clone())
  if checkpoint.opt.model == 'lstm' then
    table.insert(encoder_init_state, h_init:clone())
  end
end

function prepro(x,y)
  x = x:transpose(1,2):contiguous():resize(x:size(2), x:size(1)) -- swap the axes for faster indexing
  y = y:transpose(1,2):contiguous():resize(y:size(2), y:size(1))
  if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    y = y:float():cuda()
  end
  if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
    x = x:cl()
    y = y:cl()
  end
  return x:float(),y
end

local loader = SessionDataLoader.create(opt.test_file_path, 1, 0)
-- start sampling/argmaxing

local total_recall = 0
for i=1, loader.number_of_train_batches do

  local encoder_rnn_state = {[0] = encoder_init_state}
  local attention_input = {}
  local attention_output = {}
  -- fetch a batch
  local x, y = loader:next_train_batch()
  seq_len = y:size()[2]
  x = x:repeatTensor(checkpoint.opt.batch_size, 1)
  y = y:repeatTensor(checkpoint.opt.batch_size, 1)
  x,y = prepro(x,y)

  -- forward pass
  for t=1,seq_len do
    local encoder_lst = protos.rnn_encoder:forward{x[t], unpack(encoder_rnn_state[t-1])}
    if type(encoder_lst) ~= 'table' then 
      encoder_lst = {[1] = encoder_lst}
    end
    encoder_rnn_state[t] = {}
    for i=1,#encoder_init_state do table.insert(encoder_rnn_state[t], encoder_lst[i]) end

    table.insert(attention_input, encoder_lst[#encoder_lst])
    local c = attentions[t]:forward({encoder_lst[#encoder_lst], attention_input})
    table.insert(attention_output, c)
  end
  local decoder_rnn_state = {[0] = encoder_rnn_state[seq_len]}
  for t=1,seq_len do
    --local decoder_lst = clones.rnn_decoder[t]:forward({encoder_rnn_state[t][#init_state], unpack(decoder_rnn_state[t-1])})
    local decoder_lst = protos.rnn_decoder:forward({attention_output[t], unpack(decoder_rnn_state[t-1])})
    if type(decoder_lst) ~= 'table' then decoder_lst = {[1] = decoder_lst} end
    decoder_rnn_state[t] = {}
    for i=1,#encoder_init_state do table.insert(decoder_rnn_state[t], decoder_lst[i]) end

    local prediction = protos.rnn_projection:forward(decoder_lst[#decoder_lst]) 
    local scores, _ = prediction:topk(20, 2, true)
    local threshold, _ = scores:min(2)
    local target_scores = get_target_scores(prediction, y[t])
    if opt.gpuid ~= -1 then
      target_scores = target_scores:cuda()
    end


    local recall = (target_scores - threshold):gt(0):sum() / checkpoint.opt.batch_size
    total_recall = total_recall + recall / seq_len

    print(i)
  end
end
print(total_recall / loader.number_of_train_batches)

io.write('\n') io.flush()

