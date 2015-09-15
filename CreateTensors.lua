local CreateTensors = {}

-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

-- Generate the tensors and save to tensors directory
local function generateTensors(csv_type, csv_path, tensor_path)

	local inputFilePath  = csv_path .. csv_type ..'.csv'
	local outputFilePath = tensor_path .. csv_type ..'.th7'

	print('reading csv...')
	
	-- Count number of rows and columns in file
	local i = 0
	for line in io.lines(inputFilePath) do
	  if i == 0 then
	    COLS = #line:split(',')
	  end
	  i = i + 1
	end

	local ROWS = i - 1  -- Minus 1 because of header

	-- Read data from CSV to tensor
	local csvFile = io.open(inputFilePath, 'r')
	local header = csvFile:read()
	local data = torch.Tensor(ROWS, COLS)

	local i = 0
	for line in csvFile:lines('*l') do
	  if i % 5000 == 0 then
	  	print(' line: ', i)
	  end
	  i = i + 1
	  local l = line:split(',')
	  for key, val in ipairs(l) do
	    data[i][key] = val
	  end
	end

	csvFile:close()

	print('serializing tensor...')
	torch.save(outputFilePath, data)
end

CreateTensors.generateTensors = generateTensors

return CreateTensors