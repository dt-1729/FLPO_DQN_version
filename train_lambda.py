# train using DQN algorithm
# Write a function to learn the Neural Net Approximator for $\Lambda_{\beta, \zeta\eta}(s,a)$ for a given set of state-action parameters

import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from Env_FLPO import Env_FLPO
from Agent import lambda_approximator, train_nn, config_nn

# define function to compute target output using the training batch
def compute_Y(env:Env_FLPO, lambda_target:lambda_approximator, new_states, costs, beta, gamma):
    log_sum_exp_outs = []
    for s in new_states:
        inputs = torch.tensor([[s,*a] for a in env.get_actions(int(s))], dtype=torch.float32)
        outs = lambda_target(inputs).detach().reshape(-1)
        min_outs = torch.min(outs)
        outs_offset = outs - min_outs
        exp_outs = torch.exp(-beta/gamma*outs_offset)
        sum_exp_outs = torch.sum(exp_outs)
        log_sum_exp_outs.append(torch.log(sum_exp_outs) - beta/gamma*min_outs)
    Y = costs - 0*gamma**2/beta*np.array([tensor.item() for tensor in log_sum_exp_outs])
    return Y

# define function to update policy given the lambda_approximator neural network
def compute_policy_valueFn(env:Env_FLPO, lambda_nn:lambda_approximator, st:int, beta:float, gamma:float):
    inputs = torch.tensor([[st,*a] for a in env.get_actions(st)], dtype=torch.float32)
    lambda_st = lambda_nn(inputs).detach().reshape(-1)
    lambda_st_min = torch.min(lambda_st)
    lambda_st_offset = lambda_st - lambda_st_min
    exp_lambda_st = torch.exp(-beta/gamma*lambda_st_offset)
    sum_exp_lambda_st = torch.sum(exp_lambda_st)
    policy_st = exp_lambda_st/sum_exp_lambda_st
    V_st = -gamma/beta*(torch.log(sum_exp_lambda_st) - beta/gamma*lambda_st_min)
    return policy_st.numpy(), V_st.numpy()

# define a function to eliminate the datasets having the same input state-action pairs
def unique_tensor(dataset:list):
    # Dictionary to track unique first two elements
    unique_rows = {}
    for row in np.array(dataset):
        key = tuple(row[:2])
        if key not in unique_rows:
            unique_rows[key] = list(row)
    # Convert the dictionary values back to a NumPy array
    unique_dataset = list(unique_rows.values())
    return unique_dataset

# define a function to pick an action using a policy
def pick_act(policy_st:np.ndarray, acts:list, eps:float):
    # use epsilon-greedy strategy
    if(np.random.choice([0,1], p=[eps, 1-eps]) <= 0.5):
        # explore uniformly from the available actions
        acti = np.random.choice(range(len(acts)))
    else:
        # exploit using the given policy
        acti = np.random.choice(range(len(acts)), p=policy_st)
    return acts[acti]

# define a function to update replay memory
def update_replay_memory(replay_memory:list, transit:list):
    unique_rows = {}
    replay_memory.append(transit)
    for row in np.array(replay_memory):
        key = tuple(row[:2])
        if key not in unique_rows:
            unique_rows[key] = list(row)
    # Convert the dictionary values back to a NumPy array
    replay_memory = list(unique_rows.values())
    return replay_memory


# define a function to implement the lambda training loop
def learn_lambda(env:Env_FLPO, lambda_approx:lambda_approximator, lambda_target:lambda_approximator, train_config:dict, beta:float, gamma:float, options:dict):
    # read looping and training parameters
    steps_update_target = options['steps_update_target']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    n_episodes = options['n_episodes'] # number of episodes
    n_timeSteps = options['n_timeSteps'] # number of time steps in each episode
    eps0 = options['eps']; assert(eps0 > 0.0 or eps0 < 1.0) # exploring parameter in epsilon-greedy policy
    D_max = options['D_max']
    batch_size = options['batch_size'];
    # initialization
    eps = eps0
    replay_memory = []; assert(batch_size < D_max)
    minibatch = []

    for ep in range(n_episodes):
        # initialize a state
        st = env.state_space[np.random.choice(a=range(env.state_space_size), p=env.rho)]
        for t in range(n_timeSteps):
            # get available actions at the current state
            acts = env.get_actions(*st)
            # current policy at the current state
            policy_st, V_st = compute_policy_valueFn(env, lambda_approx, *st, beta, gamma)
            # pick an action
            act = pick_act(policy_st, acts, eps)
            # update the state using the environment probability model
            new_st = env.state_space[np.random.choice(a=range(env.state_space_size), p=env.P[(*st,*act)])]
            # obtain the corresponding cost from the environment
            cost = env.C[(*st, *act, *new_st)]
            # append the new transition to replay memory
            replay_memory = update_replay_memory(replay_memory, transit=[*st,*act,*new_st,cost])
            if len(replay_memory)>D_max:
                # remove the oldest memory if the size is full
                replay_memory.pop(0)

            # implement weight update step only when the replay memory is full
            if len(replay_memory) == D_max:
                # sample minibatch of transitions from the replay memory
                minibatch = np.array(replay_memory)[np.random.choice(a=range(len(replay_memory)), size=batch_size, replace=False)]
                sa_pairs = torch.tensor(minibatch[:,0:2], dtype=torch.float32)
                new_states = minibatch[:,2]
                costs = minibatch[:,3]
                y_target = torch.tensor(compute_Y(env, lambda_target, new_states, costs, beta, gamma), dtype=torch.float32).view(-1,1)
                # compute prediction and loss
                y_pred = lambda_approx(sa_pairs)
                loss = loss_fn(y_pred, y_target)
                # Backpropagation
                loss.backward()
                # perform weight update
                optimizer.step()
                # reset gradients
                optimizer.zero_grad()

                # update the target neural network parameters every few steps
                # if t%steps_update_target == 0:
                #     lambda_target.load_state_dict(lambda_approx.state_dict())
                #     print('target updated')
                print(f"ep {ep}, t {t}, loss {loss}, D {len(replay_memory)} \t st {st}, act {act}, new_st {new_st}, cost {cost}, eps {eps}")

            if new_st == (0,):
                # terminate the episode if the stopping state is reached
                print("----------------------------------")
                break
            else:
                # else update the state for the next time step
                st = new_st
                eps = eps**(t+1)

    return n_episodes

# define function to train the state-action value function as neural network
def learn_lambda_corrupted(env:Env_FLPO, lambda_approx:lambda_approximator, lambda_target:lambda_approximator, train_config:dict, beta:float, gamma:float, options:dict):
   
    # read the training parameters and configuration
    steps_update_target = options['steps_update_target']
    D_max = options['D_max']
    batch_size = options['batch_size']
    n_epochs = options['n_epochs']
    n_episodes = options['n_episodes']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    scheduler = train_config['scheduler']
    train_epochs = train_config['train_epochs']
    train_eps = train_config['train_eps']
    # initialization
    visits = np.zeros(shape=(env.state_space_size, env.act_space_size))
    replay_memory = []; assert(batch_size < D_max)
    episode_loss = []
    training_batch = []
    
    # perform training iterations
    for ep in range(n_episodes):
        # pick a state from the state distribution
        state = env.state_space[np.random.choice(a=range(env.state_space_size),p=env.rho)]
        episode_loss.append([])
        for epoch in range(n_epochs):
            # pick an action according to agent policy
            acts = env.get_actions(*state)
            policy_st, V_st = compute_policy_valueFn(env, lambda_approx, *state, beta, gamma)
            action = acts[np.random.choice(a=range(len(acts)), p=policy_st)]
            # increment the (state,action) pair visit count
            visits[state,action] += 1
            
            # obtain the next state from the environment
            new_state = env.state_space[np.random.choice(a=range(env.state_space_size), p=env.P[(*state,*action)])]
            # obtain the cost from the environment
            cost = env.C[(*state,*action,*new_state)]
            # append the [state,action,cost] triplet in a replay memory
            if [*state,*action,*new_state,cost] not in replay_memory:
                replay_memory.append([*state,*action,*new_state,cost])
                # remove repeated state-action pairs
                replay_memory = unique_tensor(replay_memory)
            l=len(replay_memory)
            # limit the length of the replay memory
            if l>D_max:
                replay_memory.pop(0); l -= 1; assert(l==D_max)
            # print(f'\t\treplay memory{replay_memory}')
            
            # training starts when the training batch is full
            if l == D_max:
                # sample a training batch of [s,a,c] triplets from the replay memory
                training_batch = np.array(replay_memory)[np.random.choice(a=range(l), size=batch_size, replace=False)]
                sa_pairs = torch.tensor(training_batch[:,0:2], dtype=torch.float32)
                new_states = training_batch[:,2]
                costs = training_batch[:,3]
                Y = torch.tensor(compute_Y(env, lambda_target, new_states, costs, beta, gamma), dtype=torch.float32).view(-1,1)
                loss, params_nn = train_nn(lambda_approx, loss_fn, optimizer, scheduler, sa_pairs, Y, train_epochs, train_eps)
                lambda_approx.load_state_dict(params_nn)
                break
                
            # update the target neural network parameters every few steps
            if epoch%steps_update_target == 0:
                lambda_target.load_state_dict(lambda_approx.state_dict())
            
        if len(training_batch) == batch_size:
            episode_loss.append(loss.item())
            # print(f'episode {ep}, state {state}, loss {loss}')

    return episode_loss