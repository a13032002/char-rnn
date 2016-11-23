local SessionDataLoader = {}
SessionDataLoader.__index = SessionDataLoader
local function shuffleTable(a, b, c)
  assert(#a == #b and #b == #c)
  local rand = math.random 
  local iterations = #a
  local j

  for i = iterations, 2, -1 do
    j = rand(i)
    a[i], a[j] = a[j], a[i]
    b[i], b[j] = b[j], b[i]
    c[i], c[j] = c[j], c[i]
  end
end
function SessionDataLoader.create(session_file, batch_size, train_fraction)

  local self = {}
  local x_batches = {}
  local y_batches = {}
  local number_of_nonzeros = {}
  local max_session_lengths = {}
  local global_max_session_length = 0
  local max_index = 0
  local perm_index = {}
  setmetatable(self, SessionDataLoader)

  local sessions = {}
  for line in io.lines(session_file) do
    --if #x_batches == 105 then break end
    local session = {}

    for item_id in string.gmatch(line, "%d+") do
      table.insert(session, item_id)
    end

    table.insert(sessions, session)

    if #sessions == batch_size then
      local max_session_length = 0
      local nnz = 0
      for k, session in pairs(sessions) do
        if #session > max_session_length then max_session_length = #session end
        if #session > global_max_session_length then global_max_session_length = #session end
        nnz = nnz + #session - 1
        for k, v in pairs(session) do 
          if tonumber(v) > max_index then max_index = tonumber(v) end
        end
      end

      x = torch.Tensor(#sessions, max_session_length - 1):fill(0):long()
      y = torch.Tensor(#sessions, max_session_length - 1):fill(0):long()
      for k, session in pairs(sessions) do
        padding = max_session_length - #session
        for i=1, #session-1 do
          x[{k, padding + i}] = tonumber(session[i])
          y[{k, padding + i}] = tonumber(session[i+1])
        end
      end
      table.insert(x_batches, x)
      table.insert(y_batches, y)
      table.insert(number_of_nonzeros, nnz)
      table.insert(max_session_lengths, max_session_length - 1)
      table.insert(perm_index, #x_batches)
      sessions = {}
    end
  end

  shuffleTable(x_batches, y_batches, max_session_lengths)
  number_of_batches = #x_batches
  self.number_of_train_batches = math.floor(number_of_batches * train_fraction)
  self.number_of_valid_batches = number_of_batches - self.number_of_train_batches
  self.x_batches = x_batches
  self.y_batches = y_batches
  self.number_of_nonzeros = number_of_nonzeros
  self.train_index = self.number_of_train_batches
  self.valid_index = self.number_of_valid_batches
  assert(number_of_batches > 0)
  assert(#x_batches == #y_batches and #y_batches == #max_session_lengths and #number_of_nonzeros == #y_batches)

  self.max_index = max_index
  self.max_session_lengths = max_session_lengths
  self.global_max_session_length = global_max_session_length
  self.perm_index = perm_index

  return self
end


function SessionDataLoader:next_train_batch()
  self.train_index = self.train_index + 1
  if self.train_index > self.number_of_train_batches then self.train_index = 1 end

  index = self.perm_index[self.train_index]

  return self.x_batches[index], self.y_batches[index], self.max_session_lengths[index], self.number_of_nonzeros[index]
end

function SessionDataLoader:next_valid_batch()
  self.valid_index = self.valid_index + 1
  if self.valid_index > self.number_of_valid_batches then self.valid_index = 1 end

  return self.x_batches[self.valid_index], self.y_batches[self.valid_index], self.max_session_lengths[self.valid_index], self.number_of_nonzeros[self.valid_index]
end

function SessionDataLoader:reset_valid_pointer()
  self.valid_index = 1
end

return SessionDataLoader
