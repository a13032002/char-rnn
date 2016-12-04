local SessionDataLoader = {}
SessionDataLoader.__index = SessionDataLoader
local function shuffleTable(a, b, n)
  assert(#a == #b and #a >= n)
  local rand = math.random 
  local iterations = #a
  local j

  for i = n, 2, -1 do
    j = rand(i)
    a[i], a[j] = a[j], a[i]
    b[i], b[j] = b[j], b[i]
  end
end
function SessionDataLoader.create(session_file, batch_size, valid_batches)

  local self = {}
  local x_batches = {}
  local y_batches = {}
  local global_max_session_length = 0
  local max_index = 0
  local perm_index = {}
  setmetatable(self, SessionDataLoader)

  math.randomseed(10807)
  local sessions = {}
  print("Loading Data")
  for line in io.lines(session_file) do
    --if #x_batches == 105 then break end
    local session = {}

    for item_id in string.gmatch(line, "%d+") do
      table.insert(session, item_id)
    end

    table.insert(sessions, session)
    local len = 0
    if #sessions == batch_size then
      for k, session in pairs(sessions) do
        len = #session
        for k, v in pairs(session) do 
          if tonumber(v) > max_index then max_index = tonumber(v) end
        end
      end

      if len > 30 then len = 30 end
      if len > global_max_session_length then global_max_session_length = len end
      x = torch.Tensor(#sessions, len - 1):fill(0):long()
      y = torch.Tensor(#sessions, len - 1):fill(0):long()
      for k, session in pairs(sessions) do
        for i=1, len-1 do
          x[{k, i}] = tonumber(session[i])
          y[{k, i}] = tonumber(session[i+1])
        end
      end
      table.insert(x_batches, x)
      table.insert(y_batches, y)
      table.insert(perm_index, #x_batches)
      sessions = {}
    end
  end

  number_of_batches = #x_batches
  self.number_of_valid_batches = valid_batches
  self.number_of_train_batches = number_of_batches - self.number_of_valid_batches
  shuffleTable(x_batches, y_batches, self.number_of_train_batches)
  self.x_batches = x_batches
  self.y_batches = y_batches
  self.number_of_nonzeros = number_of_nonzeros
  self.train_index = self.number_of_train_batches
  self.valid_index = self.number_of_valid_batches
  assert(number_of_batches > 0)
  assert(#x_batches == #y_batches)

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

  return self.x_batches[index], self.y_batches[index]
end

function SessionDataLoader:next_valid_batch()
  self.valid_index = self.valid_index + 1
  if self.valid_index > self.number_of_valid_batches then self.valid_index = 1 end

  return self.x_batches[self.number_of_train_batches + self.valid_index], self.y_batches[self.number_of_train_batches + self.valid_index]
end

function SessionDataLoader:reset_valid_pointer()
  self.valid_index = 1
end

return SessionDataLoader
