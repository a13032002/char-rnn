
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'rnn'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
--local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local SessionDataLoader = require 'SessionDataLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'
local LogSoftMaxProjection = require 'model.LogSoftMaxProjection'
local Attention = require 'model.Attention'

function get_target_scores(scores, y)
  assert (scores:size()[1] == y:size()[1])
  assert (scores:size()[1] == opt.batch_size)

  target_scores = torch.zeros(opt.batch_size, 1)
  for i = 1, opt.batch_size do
    target_scores[i] = scores[i][y[i]]
  end

  return target_scores
end


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'gru', 'lstm,gru or rnn')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',10,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
-- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
  local ok, cunn = pcall(require, 'clnn')
  local ok2, cutorch = pcall(require, 'cltorch')
  if not ok then print('package clnn not found!') end
  if not ok2 then print('package cltorch not found!') end
  if ok and ok2 then
    print('using OpenCL on GPU ' .. opt.gpuid .. '...')
    cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    torch.manualSeed(opt.seed)
  else
    print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
    print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- create the data loader class
--local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local loader = SessionDataLoader.create('./data/yoochoose/yoochoose-sessions.dat', opt.batch_size, 0.95)
opt.seq_length = loader.global_max_session_length
local vocab_size = loader.max_index  -- the number of distinct characters
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
  print('loading a model from checkpoint ' .. opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  protos = checkpoint.protos
  -- make sure the vocabs are the same
  local vocab_compatible = true
  local checkpoint_vocab_size = 0
  for c,i in pairs(checkpoint.vocab) do
    if not (vocab[c] == i) then
      vocab_compatible = false
    end
    checkpoint_vocab_size = checkpoint_vocab_size + 1
  end
  if not (checkpoint_vocab_size == vocab_size) then
    vocab_compatible = false
    print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
  end
  assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
  -- overwrite model settings based on checkpoint to ensure compatibility
  print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
  opt.rnn_size = checkpoint.opt.rnn_size
  opt.num_layers = checkpoint.opt.num_layers
  opt.model = checkpoint.opt.model
  do_random_init = false
else
  print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers' .. ' rnn_size ' .. opt.rnn_size)
  protos = {}
  if opt.model == 'lstm' then
    --protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
  elseif opt.model == 'gru' then
    --[[
    protos.rnn_encoder = nn.MaskZero(GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, 1), 0)
    protos.rnn_decoder = nn.MaskZero(GRU.gru(opt.rnn_size, opt.rnn_size, opt.num_layers, opt.dropout, 0), 1)
    protos.rnn_projection = nn.MaskZero(LogSoftMaxProjection.log_softmax_projection(opt.rnn_size, vocab_size), 1)
    protos.attention = nn.MaskZero(Attention.global_attention(opt.batch_size, opt.rnn_size), 1)
    --]]
    protos.rnn_encoder = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, 1)
    protos.rnn_projection = LogSoftMaxProjection.log_softmax_projection(opt.rnn_size, vocab_size)
  elseif opt.model == 'rnn' then
    --protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
  end
  --protos.criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1)
  protos.criterion = nn.CrossEntropyCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
  if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
  table.insert(init_state, h_init:clone())
  if opt.model == 'lstm' then
    table.insert(init_state, h_init:clone())
  end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
  for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
  for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn_encoder, protos.rnn_projection)

-- initialization
if do_random_init then
  params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
  for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protos.rnn.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
      end
    end
  end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
  print('cloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end
-- preprocessing helper function
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
  return x,y
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
  print('evaluating loss over split index ' .. split_index)
  local n = loader.number_of_valid_batches
  if max_batches ~= nil then n = math.min(max_batches, n) end

  loader:reset_valid_pointer(split_index) -- move batch iteration pointer for this split to front
  local loss = 0
  local encoder_rnn_state = {[0] = init_state}
  local total_recall = 0

  for i = 1,n do -- iterate over batches in the split
    -- fetch a batch
    local x, y = loader:next_valid_batch()
    seq_len = y:size()[2]
    x,y = prepro(x,y)
    -- forward pass
    for t=1,seq_len do
      clones.rnn_encoder[t]:evaluate() -- for dropout proper functioning
      local encoder_lst = clones.rnn_encoder[t]:forward{x[t], unpack(encoder_rnn_state[t-1])}
      if type(encoder_lst) ~= 'table' then 
        encoder_lst = {[1] = encoder_lst}
      end
      encoder_rnn_state[t] = {}
      for i=1,#init_state do table.insert(encoder_rnn_state[t], encoder_lst[i]) end

      local prediction = clones.rnn_projection[t]:forward(encoder_lst[#encoder_lst] )
      local scores, _ = prediction:topk(20, 2, true)
      local threshold, _ = scores:min(2)
      local target_scores = get_target_scores(prediction, y[t]):cuda()

      local recall = (target_scores - threshold):gt(0):sum() / opt.batch_size
      total_recall = total_recall + recall / seq_len

      loss = loss + clones.criterion[t]:forward(prediction, y[t]) / seq_len
    end
    -- carry over lstm state
    --rnn_state[0] = rnn_state[#rnn_state]
  end

  loss = loss / n
  total_recall = total_recall / n
  print(loss)
  print(total_recall)
  return loss
end

-- do fwd/bwd and return loss, grad_params
local encoder_init_state_global = clone_list(init_state)
local decoder_init_state_global = clone_list(init_state)
function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()

  ------------------ get minibatch -------------------
  local x, y = loader:next_train_batch()
  seq_len = y:size()[2]
  x,y = prepro(x,y)
  ------------------- forward pass -------------------
  local encoder_rnn_state = {[0] = encoder_init_state_global}
  local predictions = {}           -- softmax outputs
  local loss = 0
  for t=1,seq_len do
    -- foward encoder
    clones.rnn_encoder[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    local encoder_lst = clones.rnn_encoder[t]:forward({x[t], unpack(encoder_rnn_state[t-1])})
    encoder_rnn_state[t] = {}
    if type(encoder_lst) ~= 'table' then encoder_lst = {[1] = encoder_lst} end
    for i=1,#init_state do table.insert(encoder_rnn_state[t], encoder_lst[i]) end -- extract the state

    predictions[t] = clones.rnn_projection[t]:forward(encoder_lst[#init_state]) -- last element is the prediction
    loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
  end
  loss = loss / seq_len
  ------------------ backward pass -------------------
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local encoder_drnn_state = {[seq_len] = clone_list(init_state, true)} -- true also zeros the clones
  for t=seq_len,1,-1 do
    -- backward encoder
    local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
    local dprojection = clones.rnn_projection[t]:backward(encoder_rnn_state[t][#init_state], doutput_t)
    encoder_drnn_state[t][#init_state]:add(dprojection)
    if opt.num_layers == 1 and opt.model == 'gru'  then
      encoder_dlst = clones.rnn_encoder[t]:backward({x[t], unpack(encoder_rnn_state[t-1])}, encoder_drnn_state[t][1])
    else
      encoder_dlst = clones.rnn_encoder[t]:backward({x[t], unpack(encoder_rnn_state[t-1])}, encoder_drnn_state[t])
    end
    encoder_drnn_state[t-1] = {}
    for k,v in pairs(encoder_dlst) do
      if k > 1 then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the 
        -- derivatives of the state, starting at index 2. I know...
        encoder_drnn_state[t-1][k-1] = v
      end
    end


  end
  ------------------------ misc ----------------------
  -- transfer final state to initial state (BPTT)
  --    init_state_global = encoder_rnn_state[#encoder_rnn_state] -- NOTE: I don't think this needs to be a clone, right?
  -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
  -- clip gradient element-wise
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.number_of_train_batches
local iterations_per_epoch = loader.number_of_train_batches
local loss0 = nil
print("start training")
local total_loss = 0
for i = 1, iterations do
  local epoch = i / loader.number_of_train_batches

  local timer = torch.Timer()
  local _, loss = optim.adam(feval, params, optim_state)
  if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
    --[[
    Note on timing: The reported time can be off because the GPU is invoked async. If one
    wants to have exactly accurate timings one must call cutorch.synchronize() right here.
    I will avoid doing so by default because this can incur computational overhead.
    --]]
    cutorch.synchronize()
  end
  local time = timer:time().real

  local train_loss = loss[1] -- the loss is inside a list, pop it
  total_loss = total_loss + train_loss
  train_losses[i] = train_loss

  -- exponential learning rate decay
  if i % loader.number_of_train_batches == 0 and opt.learning_rate_decay < 1 then
    if epoch >= opt.learning_rate_decay_after then
      local decay_factor = opt.learning_rate_decay
      optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
      print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
    end
  end

  -- every now and then or on last iteration
  if i % opt.eval_val_every == 0 or i == iterations then
    -- evaluate loss on validation data
    local val_loss = eval_split(2, 100) -- 2 = validation
    val_losses[i] = val_loss

    local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.val_loss = val_loss
    checkpoint.val_losses = val_losses
    checkpoint.i = i
    checkpoint.epoch = epoch
    checkpoint.vocab = loader.vocab_mapping
    torch.save(savefile, checkpoint)
  end

  if i % opt.print_every == 0 then
    print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, total_loss / opt.print_every, grad_params:norm() / params:norm(), time))
    total_loss = 0
  end

  if i % 10 == 0 then collectgarbage() end

  -- handle early stopping if things are going really bad
  if loss[1] ~= loss[1] then
    print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
    break -- halt
  end
  if loss0 == nil then loss0 = loss[1] end
  if loss[1] > loss0 * 30 then
    print('loss is exploding, aborting.')
    break -- halt
  end
end

function range(stop)
  idx = {}
  for i = 1, stop do table.insert(idx, i) end
  return idx
end

