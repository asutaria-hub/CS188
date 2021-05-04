import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.numTrainingGames = 10000
        self.batch_size = 20

        self.l1 = nn.Parameter(self.state_size, 100)
        self.b1 = nn.Parameter(1, 100)
        self.l2 = nn.Parameter(100, 50)
        self.b2 = nn.Parameter(1, 50)
        self.l3 = nn.Parameter(50, self.num_actions)
        self.b3 = nn.Parameter(1, self.num_actions)

        self.set_weights([self.l1,
                          self.b1,
                          self.l2,
                          self.b2,
                          self.l3,
                          self.b3])


    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        out = self.run(states)
        return nn.SquareLoss(out, Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(states, self.l1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.l2)
        x = nn.AddBias(x, self.b2)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.l3)
        x = nn.AddBias(x, self.b3)
        return x

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states, Q_target)
        grad = nn.gradients(loss, self.parameters)
        for i in range(len(self.parameters)):
            self.parameters[i].update(grad[i], -self.learning_rate)
