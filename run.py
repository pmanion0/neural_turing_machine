''' Script for running RNN experiments on command line

Example:

    $ python3 run.py <model_type> <task_type> > test_out.txt

Args:
    model_type: string indicating the RNN model to use, options:
        BasicRNN, BasicLSTM, and NTM_LSTM
    
    task_type: string indicating the toy task to use, options:
        Single, StartSequence3, StartSequence20
        
Returns:
    prints periodic outputs of the average loss rate over the
    last batch of examples run through the model, e.g.
    
    $ [0] Error: tensor(0.0022, device='cuda:0', grad_fn=<DivBackward0>)
    $ [1000] Error: tensor(1.9112, device='cuda:0', grad_fn=<DivBackward0>)
'''

import torch
import sys

from torch import nn
from random import randint

from models import BasicRNN
from models import BasicLSTM
from models import NTM_LSTM
from tasks import Numbers
from training import train_model


# Read Command Line Arguments: 
model = sys.argv[1]
task = sys.argv[2]


# Number Generator + Goal Setup
max_number = 9
criterion = nn.NLLLoss()
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

setup_kwargs = {
    'train_size': 100e3,
    'number_tool': None,
    'criterion': criterion,
    'stream_size': 200,
    'model': None,
    'optim': None,
    'print_interval': 1e3
}


# Determine the Task to Train On
if task == 'Single':
    number_tool = Numbers(
        max_number,
        reset_value_func = lambda x: randint(0,max_number),
        goal_func = lambda stream: stream[0]
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
else:
    print("TASK IS NOT RECOGNIZED! ERROR!")

    
# Add Number Tool to Setup
setup_kwargs['number_tool'] = number_tool
input_dim = number_tool.get_dim()
goal_dim = number_tool.get_dim()

fake_stream = number_tool.create_stream(setup_kwargs['stream_size'])
fake_goal = number_tool.get_stream_goal(fake_stream)
goal_length = len(fake_goal)


# Determine Model + Optimization to Use
if model == 'BasicRNN':
    rnn = BasicRNN(input_dim, 5, goal_dim, output_length = goal_length)
    rnn_optim = torch.optim.SGD(rnn.parameters(), lr = 0.001)
	
    setup_kwargs['model'] = rnn
    setup_kwargs['optim'] = rnn_optim

elif model == 'BasicLSTM':
    lstm = BasicLSTM(input_dim, 5, goal_dim, output_length = goal_length)
    lstm.to(device)
	
    lstm_optim = torch.optim.SGD(lstm.parameters(), lr = 0.001, momentum = 0.9)
	
    setup_kwargs['model'] = lstm
    setup_kwargs['optim'] = lstm_optim

elif model == 'NTM_LSTM':
    memory_banks = 10
    memory_dim = 20
	
    ntm = NTM_LSTM(input_dim, 5, goal_dim, memory_banks, memory_dim, output_length = goal_length)
    ntm_optim = torch.optim.SGD(ntm.parameters(), lr = 0.001)
	
    setup_kwargs['model'] = ntm
    setup_kwargs['optim'] = ntm_optim
    setup_kwargs['print_interval'] = 1e2

else:
    print("MODEL IS NOT RECOGNIZED! ERROR!")


# Begin Training the Model!
train_model(**setup_kwargs)
