import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dp = nn.as_scalar(self.run(x))
        return 1 if dp >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        while 1:
            whole = True
            for x, y in dataset.iterate_once(batch_size):
                guess = self.get_prediction(x)
                label = nn.as_scalar(y)
                if label != guess:
                    whole = False
                    self.w.update(x, label)
            if whole:
                return

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.l1 = nn.Parameter(1, 100)
        self.b1 = nn.Parameter(1, 100)
        self.l2 = nn.Parameter(100, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.l1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.l2)
        x = nn.AddBias(x, self.b2)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        out = self.run(x)
        return nn.SquareLoss(out, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 20
        lr = 0.001
        while 1:
            div = 0
            total = 0
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                total += nn.as_scalar(loss)
                div += 1
                gl1, gb1, gl2, gb2 = nn.gradients(loss, [self.l1, self.b1, self.l2, self.b2])
                self.l1.update(gl1, -lr)
                self.b1.update(gb1, -lr)
                self.l2.update(gl2, -lr)
                self.b2.update(gb2, -lr)
            if total / div < 0.02:
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.l1 = nn.Parameter(784, 200)
        self.b1 = nn.Parameter(1, 200)
        self.l2 = nn.Parameter(200, 50)
        self.b2 = nn.Parameter(1, 50)
        self.l3 = nn.Parameter(50, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.l1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.l2)
        x = nn.AddBias(x, self.b2)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.l3)
        x = nn.AddBias(x, self.b3)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        out = self.run(x)
        return nn.SoftmaxLoss(out, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 20
        lr = 0.007
        while 1:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gl1, gb1, gl2, gb2, gl3, gb3 = nn.gradients(loss, [self.l1, self.b1, self.l2, self.b2, self.l3, self.b3])
                self.l1.update(gl1, -lr)
                self.b1.update(gb1, -lr)
                self.l2.update(gl2, -lr)
                self.b2.update(gb2, -lr)
                self.l3.update(gl3, -lr)
                self.b3.update(gb3, -lr)
            if dataset.get_validation_accuracy() > 0.98:
                return
