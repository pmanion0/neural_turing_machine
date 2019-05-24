import torch
import sys

from torch import nn
from random import randint

from models.BasicRNN import BasicRNN
from models.BasicLSTM import BasicLSTM
from models.NTM_LSTM import NTM_LSTM
from tasks.Numbers import Numbers

model = sys.argv[1]
task = sys.argv[2]

def train_model(model, number_tool, criterion, optim, train_size = 100e3, stream_size = 200, print_interval = 1e3):
    ''' Runs a full training pass for a given model '''
    
    train_size = int(train_size)
    print_interval = int(print_interval)
    error_sum = 0
    model_device = next(model.parameters()).device

    for i in range(train_size):
        # Generate a new random sequence for training
        stream = number_tool.create_stream(stream_size)
        obs = number_tool.encode_stream(stream).to(model_device)

        # Reset the model gradients and hidden layer
        model.zero_grad()
        hidden = model.init_hidden()
        
        # Score the model
        output, hidden = model.forward(obs, hidden)

        # Determine the target output and calculate the loss
        goal = number_tool.get_stream_goal(stream)
        target = torch.LongTensor([goal]).to(model_device)
        loss = criterion(output.squeeze(0), target)
        
        # Run backprop with the errors and update model/trackers
        loss.backward()
        optim.step()
        error_sum += loss

        if i % print_interval == 0:
            print('[' + str(i) + '] Error: ' + str(error_sum / print_interval))
            error_sum = 0

# Number Generator + Goal Setup
max_number = 9
goal_func = lambda stream: stream[0]
goal_dim = max_number+2

if task == 'Single':
    number_tool = Numbers(
        max_number,
        reset_value_func = lambda x: randint(0,max_number),
        goal_func = goal_func
    )

elif task == 'StartSequence3':
    number_tool = Numbers(
        max_number,
        increment_func = lambda incr: randint(0,9),
        goal_func = lambda stream: stream[0:3]
    )
elif task == 'StartSequence20':
    number_tool = Numbers(
        max_number,
        increment_func = lambda incr: randint(0,9),
        goal_func = lambda stream: stream[0:20]
    )

    
# Training Size
train_size = 100e3
criterion = nn.NLLLoss()

setup_kwargs = {
    'train_size': train_size,
    'number_tool': number_tool,
    'criterion': criterion,
    'stream_size': 200,
    'model': None,
    'optim': None
}

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

if model == 'BasicRNN':
    rnn = BasicRNN(number_tool.get_dim(), 5, goal_dim)
    rnn_optim = torch.optim.SGD(rnn.parameters(), lr = 0.001)
	
    setup_kwargs['model'] = rnn
    setup_kwargs['optim'] = rnn_optim

elif model == 'BasicLSTM':
    lstm = BasicLSTM(number_tool.get_dim(), 5, goal_dim)
    lstm.to(device)
	
    lstm_optim = torch.optim.SGD(lstm.parameters(), lr = 0.001, momentum = 0.9)
	
    setup_kwargs['model'] = lstm
    setup_kwargs['optim'] = lstm_optim

elif model == 'NTM_LSTM':
    memory_banks = 10
    memory_dim = 20
	
    ntm = NTM_LSTM(number_tool.get_dim(), 5, goal_dim, memory_banks, memory_dim)
    ntm_optim = torch.optim.SGD(ntm.parameters(), lr = 0.001)
	
    setup_kwargs['model'] = ntm
    setup_kwargs['optim'] = ntm_optim

else:
    print("MODEL IS NOT RECOGNIZED! ERROR!")

train_model(**setup_kwargs)#, print_interval = 1e2)
