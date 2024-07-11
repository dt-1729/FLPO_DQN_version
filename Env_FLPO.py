# this file defines a parametrized sequential decision making (ParaSDM) environment for the stacked FLPO problem

import numpy as np
import pickle

#define the model here as a class
class Env_FLPO():

    #define the data member for the model
    n: int; #number of nodes,
    f: int; #number of cell towers;
    nodeLocations: np.ndarray; #nx2 matrix including the node coordinates (fixed)
    facilityLocations: np.ndarray; #fx2 matrix storing the cell-tower locations (changable)
    destinationLocation: np.ndarray; #1x2 matrix storing the destination location (fixed)#
    st:int; # id for state of the system
    reward2cost:int; # reward to cost conversion factor
    selfHop:bool; # flag to indicate if self-hopping is allowed or not

    #define a probability transition matrix for the model.... (s,a) -> nextState
    P:np.ndarray;
    #define a cost function matrix for the transition (s,a) -> nextState
    C:np.ndarray;
    #define a gradient matrix wrt for each (s, a, s') for each parameters
    G:np.ndarray;
    #define a an action mask sxa dimensional matrix
    action_mask:np.ndarray;

    #define the constructor
    def __init__(self, datafile:str, init_state=0, selfHop=0, init_distribution='unifNodes'):
        '''
        function initializes the small cell network model.
        datafile: datafile containing the dataset
        init_state: initial model state (default set to state 0)
        init_distribution: initial state distribution of the model (default set of uniform distribution over the nodal states)
        '''

        #now load the mat file from the path and get the locations
        with open(datafile, 'rb') as file:
            data = pickle.load(file)
        
        # initialize model parameters
        self.n = data['numNodes'] # number of nodes
        self.f = data['numFacilities'] # number of facilities
        self.nodeLocations = data['nodeLocations']; # node locations
        self.destinationLocation = data['destinationLocation']; # destination location
        self.facilityLocations = data['facilityLocations']
        self.st = init_state; # initial state
        self.selfHop = selfHop; # whether self hopping is allowed or not

        # the state and the action space for the model
        self.state_space = [(s,) for s in np.linspace(0, self.n+self.f*self.f, self.n+self.f*self.f+1, dtype = int)];
        self.state_space_size = len(self.state_space);
        self.action_space = [(a,) for a in np.linspace(0, self.f*self.f, self.f*self.f+1, dtype = int)];
        self.act_space_size = len(self.action_space);

        self.init_action_mask(); # initialize the action-mask
        self.init_params(); #initialize model parameters, compute cost and gradient matrix
        self.init_state_distribution(init_distribution); # initialize model state distribution
        self.init_probab_matrix(); # initialze model state-transition probability
        
    # function to compute initial state distribution
    def init_state_distribution(self, distribution_type:str):
        '''
        function computes a distribution for initialization of states
        distribution_type : string to indicate whether the distribution is uniform over all the states or only over the nodes
        '''
        rho = np.zeros(self.state_space_size); # initialize

        if distribution_type == 'unifNodes':
            # uniform distribution over the nodal states
            rho[1+self.f**2:] = np.ones(self.n);
        elif distribution_type == 'unifStates':
            # uniform distribution over all the states
            rho = np.ones(self.state_space_size)

        # normalize
        self.rho = rho/np.sum(rho)

        pass;

    def init_probab_matrix(self):
        '''
        function initializes the probability transition matrix for matriced algorithms
        probability transition matrix is sxaxs dimensional tensor (3d tensor)
        '''
        # actions to states in non-lifted framework
        eps = 0.001
        P_small = eps*np.ones(shape=(self.f+1, self.f+1))
        np.fill_diagonal(P_small, np.diagonal(P_small)+(1-(self.f+1)*eps))
        Pf2f = P_small[1:self.f+1,1:self.f+1]
        Pf2d = P_small[1:self.f+1,0]

        P = np.zeros(shape = (self.state_space_size, self.act_space_size, self.state_space_size))

        for p in range(self.f+2):
            
            # state = stopping
            if p == 0:
                # only stopping state is possible
                P[p, :, 0] = np.ones(self.act_space_size) 
            # state = nodes
            elif p == self.f+1:
                # only stage 1 facilities or stopping state are possible
                P[self.f**2+1:, :, 1:self.f+1] = np.ones(shape=(self.n,self.act_space_size, self.f))
                P[self.f**2+1:, 0:self.f+1, 0:self.f+1] = P_small
            # state = final stage facilities
            elif p == self.f:
                # only stopping state is possible
                P[(p-1)*self.f+1:p*self.f+1, :, 0] = np.ones(shape=(self.f,self.act_space_size))
            # state = intermediate facilities
            elif p > 0 and p <self.f:
                # matrix to prevent self hopping
                # noSelfHoppingMatrix = np.zeros(shape=(self.A,self.f))
                # noSelfHoppingMatrix[(p-1)*self.f:(p-1)*self.f+self.f, :] = np.eye(self.f)
                # only next stopping state and stage facilities are possible
                P[(p-1)*self.f+1:p*self.f+1, :, p*self.f+1:(p+1)*self.f+1] = np.ones(shape=(self.f,self.act_space_size,self.f)) #- (1-self.selfHop)*noSelfHoppingMatrix
                P[(p-1)*self.f+1:p*self.f+1, 0, :] = np.eye(1,self.f**2+self.n+1) # for action = go-to-stopping-state
                # P[(p-1)*self.f+1:p*self.f+1, 0, :] = Pf2d
                P[(p-1)*self.f+1:p*self.f+1, p*self.f+1:(p+1)*self.f+1, p*self.f+1:(p+1)*self.f+1] = Pf2f
                if self.selfHop == 0:
                    P[(p-1)*self.f+1:p*self.f+1, p*self.f+1, p*self.f+1] = 0.0 # prevent self hopping
        # normalize the distribution
        self.P = P/np.sum(P,axis=2,keepdims=True)

        pass

    #define a function to initialize the probability transition
    # def init_probab_matrix(self): ##Exclusive to the matricized version of the algorithm
    #     '''
    #     function initializes the probability transition matrix for matriced algorithms
    #     probability transition matrix is sxaxs dimensional tensor (3d tensor)
    #     '''
    #     # make p(s'|s,a) = 0 for all a if transition s --> s' is not possible
    #     eps = 0.001
    #     P_small = eps*np.ones(shape=(self.f+1, self.f+1))
    #     np.fill_diagonal(P_small, np.diagonal(P_small)+(1-(self.f+1)*eps))
    #     P = np.zeros(shape=(self.act_space_size, self.state_space_size))
        
    #     Pf2f = P_small[1:self.f+1,1:self.f+1]
    #     Pf2d = P_small[1:self.f+1,0]

    #     # now fill the higher dimensional probability matrix with the values of intermediate costs for allowable transitions
    #     for p in range(self.f+1):
    #         if p == 0:
    #             # from destination
    #             P[0,0] = 1; # to destination only
    #         elif p == 1:
    #             # from FLPO stage 1 facilities
    #             P[1:self.f+1, 0] = Pf2d; # to destination 
    #             P[1:self.f+1, 1:1+self.f] = Pf2f; # to FLPO stage 2 facilities
    #         elif p <= self.f and p > 1:
    #             # from FLPO stage p facilities
    #             P[1+(p-1)*self.f:1+p*self.f, 0] = Pf2d # to destination
    #             P[1+(p-1)*self.f:1+p*self.f, 1+(p-1)*self.f:1+p*self.f] = Pf2f # to FLPO stage p+1 facilities

    #     self.P = np.tile(np.expand_dims(P, axis=0), (self.state_space_size,1,1));
        
    #     pass;
    
    #define a function to update cost matrix..after change of parameters
    def compute_cost_matrix(self):
        penalty = 1000
        # initialize the cost matrix at very high cost for all state-action pairs
        C = penalty*np.ones(shape = (self.state_space_size, self.state_space_size))
        # compute lower dimensional cost matrix of the original FLPO problem as intermediate cost
        C_interm = np.sum(self.X**2, axis=1, keepdims=True) + np.sum(self.X**2, axis=1, keepdims=True).T - 2*(self.X @ self.X.T);
        # normalization_const = np.max(C_interm)
        
        # pick sections of intermediate cost
        Cf2f = C_interm[1:self.f+1, 1:self.f+1] + (1-self.selfHop)*1000*np.eye(self.f); # facilities to facilities, also add self Hopping penalty
        Cf2d = C_interm[1:self.f+1, 0]; # facilities to destination
        Cn2f = C_interm[self.f+1:, 1:self.f+1]; # nodes to facilities
        Cn2d = C_interm[self.f+1:, 0] # nodes to destination

        # now fill the higher dimensional cost matrix with the values of intermediate costs for allowable transitions
        for p in range(self.f+2):
            if p == 0:
                # from destination
                C[0,0] = 0 # to destination only
            elif p == 1:
                # from FLPO stage 1 facilities
                C[1:self.f+1, 0] = Cf2d; # to destination
                C[1:self.f+1, self.f+1:1+2*self.f] = Cf2f; # to FLPO stage 2 facilities
            elif p < self.f and p > 1:
                # from FLPO stage p facilities
                C[1+(p-1)*self.f:1+p*self.f, 0] = Cf2d # to destination
                C[1+(p-1)*self.f:1+p*self.f, 1+p*self.f:1+(p+1)*self.f] = Cf2f # to FLPO stage p+1 facilities
            elif p == self.f:
                # from FLPO final stage facilities
                C[1+(p-1)*self.f:1+p*self.f, 0] = Cf2d # to destination only
            elif p == self.f+1:
                # from FLPO nodes
                C[1+(p-1)*self.f:1+(p-1)*self.f+self.n, 1:1+self.f] = Cn2f # to facilities
                C[1+(p-1)*self.f:1+(p-1)*self.f+self.n, 0] = Cn2d # to destination
        
        # expand the cost matrix to define it for each action as well
        self.C = np.tile(np.expand_dims(C, axis=1), (1, self.act_space_size, 1))
        # self.C = np.ones(shape = (self.state_space_size,self.act_space_size,self.state_space_size))

        # penalize unavailable actions
        for st in range(len(self.C)):
            for act in range(len(self.C[st])):
                if self.action_mask[st,act] == 0:
                    self.C[st,act,:] = penalty
        
        # self.C = self.C/normalization_const
        
        pass

    #define a function to iniitalize the gradient wrt parameters for each sxaxs combintation
    def compute_grad_matrix(self):
        '''
        initialize the gradient w.r.t parameter tensor which is a state x action x nextState x parameter dimensional tensor..
        '''
        # initialize lower dimensional gradient matrix
        Gtemp = np.zeros(shape = (self.f+self.n+1, self.f+self.n+1, len(self.parameters())));
        # initialize higher dimensional gradient matrix
        G = np.zeros(shape = (self.state_space_size, self.state_space_size, len(self.parameters())));

        # fill up the values for the lower dimensional gradient matrix
        for p in range(self.f):
            Gtemp[p+1,:,2*p] = (self.X[p+1,0] - self.X[:,0])*2;
            Gtemp[p+1,:,(2*p)+1] = (self.X[p+1,1] - self.X[:,1])*2;
            Gtemp[:,p+1,2*p] = (self.X[p+1,0] - self.X[:,0])*2;
            Gtemp[:,p+1,(2*p)+1] = (self.X[p+1,1] - self.X[:,1])*2;
        Gd2f = Gtemp[0,1:self.f+1,:]; # destination to facilities 
        Gf2f = Gtemp[1:self.f+1,1:self.f+1,:]; # facilities to facilities
        Gf2n = Gtemp[1:self.f+1,self.f+1:,:]; # facilities to nodes
        Gf2d = Gtemp[1:self.f+1,0,:]; # facilities to destination
        Gn2f = Gtemp[self.f+1:,1:self.f+1,:]; # nodes to facilities

        G[0, 1:1+self.f*self.f, :] = np.tile(Gd2f, (self.f,1)); # destination 2 facilities
        G[1:1+self.f*self.f, 0, :] = np.tile(Gf2d, (self.f,1)); # facilities 2 destination
        G[1:1+self.f*self.f, 1:1+self.f*self.f,:] = np.tile(Gf2f, (self.f,self.f,1)); # facilities 2 facilities
        G[1:1+self.f*self.f, 1+self.f*self.f:,:] = np.tile(Gf2n, (self.f,1,1)); # facilities 2 nodes
        G[1+self.f*self.f:, 1:1+self.f*self.f,:] = np.tile(Gn2f, (1,self.f,1)); # nodes 2 facilities

        #now populate the required elements of the matrix
        self.G = np.tile(np.expand_dims(G, axis=1), (1, self.act_space_size, 1, 1));
        pass;
    
    #define a function to initialize the parameters for the model
    def init_params(self):
        ''' 
        function initializes model parameters, computes the state-cost transition and its gradient with respect to parameters
        '''
        #initialize the model parametrs to all zeros
        # self.facilityLocations = np.zeros(shape = (self.f, 2));
        self.X = np.vstack([self.destinationLocation, self.facilityLocations, self.nodeLocations]);
        
        self.compute_cost_matrix();
        # self.compute_grad_matrix();
        pass;
    
    #define a function to get the parameters for the model
    def parameters(self) -> np.ndarray:

        '''
        function returns a list of parameters for the system.
        '''

        #return the cell locations as the parameters list. each coordinate is a scalar parameter..
        return self.facilityLocations.reshape(-1);
    
    #define a function to update the parameters to a new value
    def update_params(self, params: np.ndarray):
        '''
        function to update the list of paramaters for the model to a new set of parameters
        params: updated list f parameters of the same length and shape of elements as in self.parameters()
        return :
        None
        '''

        #firstly check if the number of parameters in the list match
        assert(params.shape == self.parameters().shape);

        #now go over each parameter in the list and update the corresponding model parameter
        # for i in range(len(params)):
        #     #assign the correspondin parameter after checking the dimension
        #     # in this case of small cell network every parameter element is a coordinate in the facilityLocations matrix.
        #     # assert(params[i].shape == 1);
        #     self.facilityLocations[int(i/2), i%2] = params[i];
        
        self.facilityLocations = params.reshape(-1,2);

        #before updating the cost and gradient tensors...update the self.X vector with the updated parameters.
        self.X[1:self.f+1,:] = self.facilityLocations;

        #once parameters are updated...update the cost and gradient tensors for the model..
        self.compute_cost_matrix();
        # self.compute_grad_matrix();

        pass;
    
    #define a function to set the system state
    def setState(self, state):
        #just set the state
        assert((state <= self.n + self.f*self.f) & (state>=0));
        self.st = state;
        pass;
    
    def state(self):
        return (self.st,);
    
    def get_actions(self, st):
        '''
        this function gives the set of all possible actions at a given state
        st: given state index lying in {0, 1, 2, . . . , 1+n+f^2}
        return:
        action_list: list of actions available for the present state of the system
        '''
        # check if the state st is a valid state
        assert((st <= self.n + self.f*self.f) & (st>=0));

        actions = [(a,) for a in np.where(self.action_mask[st] == 1)[0]]

        return actions
    
    #define a functino to define the action mask for the model....for matricized implementation
    def init_action_mask(self):
        '''
        Function to indicate allowable actions at every state
        '''
        self.action_mask = np.zeros(shape = (self.state_space_size, self.act_space_size));

        # allowable actions from destinations
        self.action_mask[0,0] = 1 # only destination is possible
        # allowable actions from nodes
        self.action_mask[self.f*self.f+1:, 0:self.f+1] = np.ones((self.n, 1+self.f)) # only destination and stage 1 facility locations are possible
        # allowable actions from stage p to stage (p+1) facilities
        for p in range(1,self.f):
            #only stage p to stage p+1 allowed (no self hopping) for p < self.f
            self.action_mask[(p-1)*self.f+1:p*self.f+1, p*self.f+1:(p+1)*self.f+1] = np.ones((self.f,self.f)) #- np.eye(self.f)*(1-self.selfHop)
        self.action_mask[1:self.f*self.f+1,0] = 1 # destination allowed from all facilities in all stages

        pass

