local Attention = {}

function Attention.global_attention(batch_size, hidden_size)
  --[[
    inpits[1]: a hidden state (batch_size, hidden_size)
    inputs[2]: a table containing previous hidden states (batch_size, hidden_size)

    output[1]: output context vector that is a linear combination of input hidden states
  --]]
  --
  
  local current_hidden = nn.Identity()()
  local previous_hidden_states = nn.JoinTable(2)()

  local hidden_matrices = nn.View(batch_size, -1, hidden_size):setNumInputDims(2)(previous_hidden_states) -- (batch_size, number_of_previous_step, hidden_size)
  local current_hidden_matrices = nn.View(batch_size, 1, hidden_size)(current_hidden)
  local score = nn.MM(false, true)({hidden_matrices, current_hidden_matrices})
  local a = nn.SoftMax()(nn.View(batch_size, -1):setNumInputDims(3)(score))
  local output = nn.MixtureTable(2)({a, hidden_matrices})

  return nn.gModule({current_hidden, previous_hidden_states}, {output})
end

return Attention
