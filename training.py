import torch

def train_model(model, number_tool, criterion, optim, train_size = 100e3, stream_size = 200, print_interval = 1e3):
    ''' Runs a full training routine for a given model '''
    
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
        target = torch.LongTensor(goal).to(model_device)
        loss = criterion(output.squeeze(1), target)

        # Run backprop with the errors and update model/trackers
        loss.backward()
        optim.step()
        error_sum += loss

        if i % print_interval == 0:
            print('[' + str(i) + '] Error: ' + str(error_sum / print_interval))
            error_sum = 0