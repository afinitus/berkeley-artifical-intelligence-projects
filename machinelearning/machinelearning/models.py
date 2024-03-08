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
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        #tells if vector product is greater than or below 0, in acordance with prediction
        if nn.as_scalar(self.run(x)) < 0:
            return -1
        return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        #at the start our dataset is not converged
        converged = False
        while not converged:
            #assume the dataset has converged
            converged = True
            data = dataset.iterate_once(1)
            for x,y in data:
                #we iterate through the data and check if the values match the predictions
                #if they dont then we have not yet converged and we continue iterating
                if nn.as_scalar(y) != self.get_prediction(x):
                    self.w.update(x, nn.as_scalar(y))
                    converged = False

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        #these are appropriate parameters according to the batchsize we are using and the question
        self.w1 = nn.Parameter(1, 200)
        self.w2 = nn.Parameter(200, 1)
        self.b1 = nn.Parameter(1, 200)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        #we basically just do the operation that is listed for f(x), which is do a linear transformation
        #on weight vector 1 by x
        #next we want to do the relu on this new vector added with out b1 vector
        #finally we want to multiply with wight vector 2, and add b2 to the whole thing
        new_x = nn.Linear(x, self.w1)
        new_x_and_b = nn.AddBias(new_x, self.b1)
        relu_new_x_b = nn.ReLU(new_x_and_b)
        first_term = nn.Linear(relu_new_x_b , self.w2)
        f = nn.AddBias(first_term, self.b2)
        return f

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        #we train this model the same way we would the perceptron
        converged = False
        while not converged:
            #using a different batch size of 200
            data = dataset.iterate_once(200)
            for x,y in data:
                #to train first we need to get the parameter gradients, and then update each parameter accordingly
                update_directions = nn.gradients(self.get_loss(x, y), [self.w1, self.w2, self.b1, self.b2])
                #using the provided .05 learning rate
                self.w1.update(update_directions[0], -0.05)
                self.w2.update(update_directions[1], -0.05)
                self.b1.update(update_directions[2], -0.05)
                self.b2.update(update_directions[3], -0.05)
                #we want a max loss of .02 to make the training model as strong as possible
                #so as soon as we reach this we will call this the convergence barrier, and stop updating
            current_loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if (nn.as_scalar(current_loss) < 0.02):
                converged = True
                break 

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
        #these are appropriate parameters according to the batchsize we are using and the question
        self.w1 = nn.Parameter(784, 200)
        self.w2 = nn.Parameter(200, 10)
        self.b1 = nn.Parameter(1, 200)
        self.b2 = nn.Parameter(1, 10)

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
        #same code as q2
        new_x = nn.Linear(x, self.w1)
        new_x_and_b = nn.AddBias(new_x, self.b1)
        relu_new_x_b = nn.ReLU(new_x_and_b)
        first_term = nn.Linear(relu_new_x_b , self.w2)
        f = nn.AddBias(first_term, self.b2)
        return f

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
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        #we pretty much follow the same code as in q2 but with minor changes
        converged = False
        while not converged:
            #we iterate with the batchsize which we are told in the question is 100
            data = dataset.iterate_once(200)
            for x, y in data:
                #we will iterate through the data and update the w and b vectors accordingly with the loss gradient
                update_directions = nn.gradients(self.get_loss(x, y), [self.w1, self.w2, self.b1, self.b2])
                #experimentally we are using the same values as in question 2 for the multiplier
                self.w1.update(update_directions[0], -0.5)
                self.w2.update(update_directions[1], -0.5)
                self.b1.update(update_directions[2], -0.5)
                self.b2.update(update_directions[3], -0.5)
            #the question statement says we need an accuracy of at least 97% so .97 is our value
            if dataset.get_validation_accuracy() >= 0.97:
                converged = True
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
