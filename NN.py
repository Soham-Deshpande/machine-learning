import numpy as np

# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
y = np.array(([0], [1], [1], [0]), dtype=float)


# Define useful functions

# Activation function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


NN = NeuralNetwork(X, y)
for i in range(1500):  # trains the NN 1,500 times
    if i % 500 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("\n")

    NN.train(X, y)


def do_rmsprop():
    w, b, eta = init_w, init_b, 0.1
    v_w, v_b, beta, eps = 0, 0, 0.9, 1e-8
    for i in range(max_epochs):
        dw, db = 0, 0
        for x,y in zip(X,Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        v_w = beta * v_w + (1 - beta) * dw**2
        v_b = beta * v_b + (1 - beta) * db**2

        w = w - (eta/np.sqrt(v_w + eps)) * dw
        b = b - (eta/np.sqrt(v_b + eps)) * db


def do_adam():
    w, b, eta, max_epochs = 1, 1, 0.01, 100,
    m_w, m_b, v_w, v_b, eps, beta1, beta2 = 0, 0, 0, 0, 1e-8, 0.9, 0.99
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in data:
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        m_w = beta1 * m_w + (1 - beta1) * dw
        m_b = beta1 * m_b + (1 - beta1) * db

        v_w = beta2 * v_w + (1 - beta2) * dw ** 2
        v_b = beta2 * v_b + (1 - beta2) * db ** 2

        m_w = m_w / (1 - beta1 ** (i + 1))
        m_b = m_b / (1 - beta1 ** (i + 1))

        v_w = v_w / (1 - beta2 ** (i + 1))
        v_b = v_b / (1 - beta2 ** (i + 1))

        w = w - eta * m_w / np.sqrt(v_w + eps)
        b = b - eta * m_b / np.sqrt(v_b + eps)
    print(error(w, b))



def vae_loss(output, input, mean, logvar, loss_func):
    recon_loss = loss_func(output, input)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mean**2 - 1. - logvar, 1))
    return recon_loss + kl_loss


class VAE(nn.Module):
    def __init__(self, in_shape, n_latent):
        super().__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w = in_shape
        self.z_dim = h//2**2 # receptive field downsampled 2 times
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # 32, 16, 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32, 8, 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.z_mean = nn.Linear(64 * self.z_dim**2, n_latent)
        self.z_var = nn.Linear(64 * self.z_dim**2, n_latent)
        self.z_develop = nn.Linear(n_latent, 64 * self.z_dim**2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
            CenterCrop(h,w),
            nn.Sigmoid()
        )

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size()))
        return (noise * stddev) + mean

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar
