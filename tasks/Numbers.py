import torch

class Numbers:
    ''' This is a toy task that asks for a single number return
    
    '''
    def __init__(self, max_number = 9,
                 increment_func = lambda stream: stream[-1]+1,
                 reset_func = None,
                 reset_value_func = lambda stream: 0,
                 goal_func = lambda stream: stream[0]):
        ''' Initialize the controller
        Args:
            increment_func: called on each number in the stream to decide
                the next number in the stream
            reset_func: evalutes the stream to determine if a reset
                condition has been met causing a restart on a new number
            reset_value_func: if a reset is triggered, this is called to
                determine the new value to restart the stream with
            goal_func: a function that returns the target output value
                for the stream (i.e. the goal output)
        '''
        self.max_number = max_number
        self.invalid_token = 'INVALID'
        self.end_token = 'END'
        self.all_numbers = list(range(max_number+1))
        self.all_classes = self.all_numbers + [self.invalid_token, self.end_token]
        
        self.increment_func = increment_func
        self.reset_func = reset_func
        self.reset_value_func = reset_value_func
        self.goal_func = goal_func
        
        
    def create_stream(self, length, seed_stream = [0]):
        ''' Create a stream (sequence) of numbers
        
        Args:
            length: length of the stream to generate
            seed_stream: seed stream to use in generating the first number
                
        Returns:
            A sequence (list) of numbers in the stream
        '''
        if self.reset_func == None:
            self.reset_func = lambda stream: stream[-1] == self.max_number
    
        start_num = self.reset_value_func(seed_stream)
        stream = [start_num]
        
        for i in range(max(0,length-1)):
            if self.reset_func(stream):
                new_num = self.reset_value_func(stream)
            else:
                new_num = self.increment_func(stream)
            stream.append(new_num)
            
        stream.append(self.end_token)
            
        return stream
    
    
    def encode_stream(self, num_stream):
        ''' Converts number or number_stream into one-hot encoded tensor
        An additional category is added for OTHER (unrecognized numbers)
        
        Args:
            num_stream: A single number of sequence (list) of numbers
                that will be encoded
            
        Returns:
            Encoded tensor of shape (number_index, _, one_hot_encoded_number)
        '''
        num_stream = num_stream if type(num_stream) is list else [num_stream]
        output = torch.zeros(len(num_stream), 1, len(self.all_classes))
        
        for i, num in enumerate(num_stream):
            value = num if num in self.all_numbers else self.invalid_token
            ind = self.all_classes.index(value)
            output[i,0,ind] = 1
            
        return output
    
    def get_stream_goal(self, stream):
        ''' Return the goal output for the providede stream 
        Args:
            stream: a list of numbers in the sequence
        
        Returns:
            Simple evaluation of the goal function (type depends
            on the goal function provided)
        '''
        goal = self.goal_func(stream)
        return goal if type(goal) is list else [goal]
    
    
    def decode_row(self, encoded_number):
        ''' Decode a one-hot encoded vector back to the number '''
        ind = encoded_number.argmax()
        value = self.all_numbers[ind] if ind in self.all_numbers else -1
        return value
    
        
    def get_dim(self):
        ''' Return dimension of each encoded tensor '''
        return len(self.all_classes)