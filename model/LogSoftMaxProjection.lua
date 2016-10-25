local LogSoftMaxProjection = {}

function LogSoftMaxProjection.log_softmax_projection(input_size, output_size)
  local inputs = {}

  table.insert(inputs, nn.Identity()()) -- x

  local projection = nn.Linear(input_size, output_size)(inputs[1])
  local output = nn.LogSoftMax()(projection)

  local outputs = {}

  table.insert(outputs, output)

  return nn.gModule(inputs, outputs)
end

return LogSoftMaxProjection
