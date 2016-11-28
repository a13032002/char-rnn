local PositiveNegativeDifference, Parent = torch.class('nn.PositiveNegativeDifference', 'nn.Module')

function PositiveNegativeDifference:__init()
  Parent.__init(self)
  self.Sigmoid = nn.Sigmoid()
  self.Log = nn.Log()
  self.CSubTable = nn.CSubTable()
end

function PositiveNegativeDifference:updateOutput(input)
  local scores, target = unpack(input)
  local input_size = scores:size()[2]
  local batch_size = scores:size()[1]
  local positive = scores:index(2, target:long()):diag():resize(batch_size, 1):repeatTensor(1, input_size)
  self.positive = positive
  local difference = self.CSubTable:forward({positive, scores})
  self.output = difference

  return self.output
end

function PositiveNegativeDifference:updateGradInput(input, gradOutput)
  local scores, target = unpack(input)
  local input_size = scores:size()[2]
  local batch_size = scores:size()[1]

  local grad_positve, grad_input = unpack(self.CSubTable:backward({self.positive, scores}, gradOutput))
  
  grad_positve = grad_positve:sum(2)
  for i = 1, batch_size do
    grad_input[i][target[i]] = grad_input[i][target[i]] + grad_positve[i]
  end

  self.gradInput = grad_input

  return self.gradInput
end

function PositiveNegativeDifference:accGradParameters(input, gradOutput)
end

function PositiveNegativeDifference:reset()
end
