require 'nn'
local RankingCriterion, parent = torch.class('nn.RankingCriterion', 'nn.Criterion')
function RankingCriterion:__init()
  parent.__init(self)
end


function RankingCriterion:updateOutput(input, target)
  local input_size = input:size()[2]
  local batch_size = input:size()[1]
  local positive = input:index(2, target:long()):diag():resize(batch_size, 1):repeatTensor(1, input_size)
  self.positive = positive
  local difference = self.CSubTable:forward({positive, input})
  self.difference = difference
  local sigmoid_values = nn.Sigmoid:forward(difference)
  self.sigmoid_values = sigmoid_values
  local log_sigmoid_values = self.Log:forward(sigmoid_values)

  return log_sigmoid_values:mean()
end

function RankingCriterion:updateGradInput(input, target)
  local batch_size = input:size()[1]
  local input_size = input:size()[2]
  local grad = torch.ones(batch_size, input_size) / batch_size / input_size

  grad = self.Log:backward(self.sigmoid_values, grad)
  print(grad:size())
  print(self.difference:size())
  grad = self.Sigmoid:backward(self.difference, grad)
  local grad_positve, grad_input = unpack(self.CSubTable:backward({self.positive, input}, grad))
  
  grad_positve = grad_positve:sum(2)
  for i = 1, batch_size do
    grad_input[i][target[i]] = grad_input[i][target[i]] + grad_positve[i]
  end

  return grad_input
end
