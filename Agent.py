import torch 
import torch.nn as nn 
import torch.optim as optim

### Define a neural network model
#Structure:
#Input layer: 2 nodes corresponding to state-action pair $(s,a) \in \mathcal{S}\times\mathcal{A}$
#Output layer: 1 node corresponding to the vector value $\Lambda(s,a)$
#We initialize two neural networks, an approximator DNN: $\hat{\Lambda}(s,a,w)$ and a target DNN: $\Lambda^t(s,a,w)$

# define agent neural network class
class lambda_approximator(nn.Module):
    
    def __init__(self):
        super().__init__()
        # first hidden layer; 2 inputs (from input layer), 10 outputs
        self.hidden1 = nn.Linear(2,50)
        # first relu activation layer
        self.act1 = nn.ReLU()
        # # second hidden layer; 10 inputs, 8 outputs
        # self.hidden2 = nn.Linear(10,8)
        # # # second relu activation layer 
        # self.act2 = nn.ReLU()
        # # third hidden layer; 8 inputs, 4 outputs
        # self.hidden3 = nn.Linear(8,4)
        # # third relu activation layer
        # self.act3 = nn.ReLU()
        # output layer; 8 inputs, 1 output
        self.output = nn.Linear(50,1)

    def forward(self, x):
        x = self.hidden1(x) # pass input through first hidden layer
        x = self.act1(x) # through first activation layer
        # x = self.hidden2(x) # through second hidden layer
        # x = self.act2(x) # through second activation layer
        # x = self.hidden3(x) # through third hidden layer
        # x = self.act3(x) # through third activation layer
        x = self.output(x) # through output layer
        return x
    
# define a training function for 1 epoch
def train_nn(model_nn, loss_fn, optimizer, scheduler, x_train, y_target, epochs, episodes):
    assert(len(x_train) > epochs)
    batch_size = int(len(x_train)/epochs)
    print('\t\t\t inside training')
    print('\t\t\t training data >>>>\n', 'input =', x_train, 'target =', y_target)
    for ep in range(episodes):
        for i in range(epochs):
            x_batch = x_train[i:i+batch_size]
            y_pred = model_nn(x_batch)
            ybatch = y_target[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'\t\t\t training ep {ep}, loss {loss}')
        scheduler.step()
    return loss, model_nn.state_dict()

# define the training configuration
def config_nn(lambda_approx:lambda_approximator, optimizer_name:str, lr:float, momentum=0.99, gamma=0.9):
    loss_fn = nn.MSELoss() # mean square loss
    if optimizer_name == 'adam':
        optimizer = optim.Adam(lambda_approx.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(lambda_approx.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(lambda_approx.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    return loss_fn, optimizer, scheduler