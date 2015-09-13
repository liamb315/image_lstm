-- Loader for image-level models

require 'torch'
require 'math'

local ImageLoader = {}
--[[TODO: Learn about the __index metamethod, reaccesses table
        http://www.lua.org/pil/13.4.1.html--]]
ImageLoader.__index = ImageLoader  


function ImageLoader.create(tensor_x_file, tensor_y_file, batch_size, seq_length, input_size)
	local self = {}
	setmetatable(self, ImageLoader)

	self.batch_size = batch_size
	self.seq_length = seq_length
	
	local x_data = torch.load(tensor_x_file)
	local y_data = torch.load(tensor_y_file)

	-- Cut off the end so that it divides evenly
    local len = x_data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        x_data = x_data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
        y_data = y_data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end

	-- Consistency checks on length of x_data, y_data
	assert(x_data:size(1) == y_data:size(1))

	self.nbatches          = x_data:size(1)/(batch_size*seq_length)
	
	local num_images       = x_data:size(1)
	local num_properties   = num_images/seq_length --Assumes that each property has seq_length images

	print('reshaping tensors...')
	self.x_data            = x_data:reshape(num_properties, seq_length, input_size)
	self.y_data            = y_data:reshape(num_properties, seq_length)
	self.current_batch     = 0
	self.evaluated_batches = 0 -- number of times next_batch() called

	print('data load complete.')
	collectgarbage()
	return self
end

-- Return all the data
function ImageLoader:load_image_data()
	return self.x_data, self.y_data
end


-- Return a batch 
function ImageLoader:next_image_batch()
	-- Return to beginning of the dataset after exhausting examples
	if self.current_batch == self.nbatches then
		self.current_batch = 0
	end

	local lo = self.current_batch * self.batch_size + 1
	local hi = lo + self.batch_size - 1

	self.current_batch     = self.current_batch + 1
	self.evaluated_batches = self.evaluated_batches + 1

	return self.x_data[{{lo, hi}, }], self.y_data[{{lo, hi}, }]
end


return ImageLoader