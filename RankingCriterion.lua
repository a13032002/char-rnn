local RankingCriterion, parent = torch.class('nn.RankingCriterion', 'nn.Criterion')
function RankingCriterion:__init()
  Criterion.__init(self)
  self.sigmoid = nn.Sigmoid()
  self.log = nn.Log()
end


function RankingCriterion:updateOutput(input, target)
  local input_size = input:size()[2]
  local positive = input:index(2, target:long()):diag():repeatTensor(1, input_size)
  local difference = positive - input
  local sigmoid_values = nn.Sigmoid:forward(difference)
  local log_sigmoid_values = nn.Log:forward(sigmoid_values)


  
