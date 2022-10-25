from sacred import Experiment
from sacred.observers import FileStorageObserver

import torch
import torch.nn.functional as F
import numpy as np
import math

if torch.cuda.is_available():
    torch.cuda.set_device(0)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / np.sqrt(in_dim / 2.0)
    return np.random.normal(size=size, scale=xavier_stddev)


ex = Experiment("train")
ex.observers.append(FileStorageObserver.create("Experiments"))


@ex.config
def config():
    # System Parameters
    dataset = "Letter"  # Dataset
    repetition = 10  # repeat the experiment a couple of times
    iteration = 5000  # Number of iteration
    batch_size = 128  # Mini batch size
    mr = 0.5  # Missing rate
    hr = 0.8  # Hint rate
    alpha = 10  # Loss Hyperparameters
    train_rate = 0.8  # Train Rate
    is_fi = False  # Use fake indication
    is_dh = False  # Use dynamic hint rate


@ex.automain
def main(
    _run,
    dataset,
    repetition,
    iteration,
    batch_size,
    mr,
    hr,
    alpha,
    train_rate,
    is_fi,
    is_dh,
):
    # Data generation
    Data = np.loadtxt(".\\Datasets\\" + dataset + ".csv", delimiter=",", skiprows=1)

    # Parameters
    No = len(Data)
    Dim = len(Data[0, :])

    # Hidden state dimensions
    H_Dim1 = Dim
    H_Dim2 = Dim

    # Normalization (0 to 1)
    for i in range(Dim):
        Data[:, i] = Data[:, i] - np.min(Data[:, i])
        Data[:, i] = Data[:, i] / (np.max(Data[:, i]) + 1e-6)

    # Missing introducing
    mr_vec = mr * np.ones((Dim, 1))

    Missing = np.zeros((No, Dim))

    for i in range(Dim):
        A = np.random.uniform(0.0, 1.0, size=[len(Data),])
        B = A > mr_vec[i]
        Missing[:, i] = 1.0 * B

    # Train Test Division
    idx = np.random.permutation(No)

    Train_No = int(No * train_rate)
    Test_No = No - Train_No

    # Train / Test Features
    trainX = Data[idx[:Train_No], :]
    testX = Data[idx[Train_No:], :]

    # Train / Test Missing Indicators
    trainM = Missing[idx[:Train_No], :]
    testM = Missing[idx[Train_No:], :]

    # 1. Discriminator
    if torch.cuda.is_available():
        D_W1 = torch.tensor(
            xavier_init([Dim * 2, H_Dim1]), requires_grad=True, device="cuda"
        )  # Data + Hint as inputs
        D_b1 = torch.tensor(np.zeros(shape=[H_Dim1]), requires_grad=True, device="cuda")

        D_W2 = torch.tensor(
            xavier_init([H_Dim1, H_Dim2]), requires_grad=True, device="cuda"
        )
        D_b2 = torch.tensor(np.zeros(shape=[H_Dim2]), requires_grad=True, device="cuda")

        D_W3 = torch.tensor(
            xavier_init([H_Dim2, Dim]), requires_grad=True, device="cuda"
        )
        D_b3 = torch.tensor(
            np.zeros(shape=[Dim]), requires_grad=True, device="cuda"
        )  # Output is multi-variate
    else:
        D_W1 = torch.tensor(
            xavier_init([Dim * 2, H_Dim1]), requires_grad=True
        )  # Data + Hint as inputs
        D_b1 = torch.tensor(np.zeros(shape=[H_Dim1]), requires_grad=True)

        D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]), requires_grad=True)
        D_b2 = torch.tensor(np.zeros(shape=[H_Dim2]), requires_grad=True)

        D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]), requires_grad=True)
        D_b3 = torch.tensor(
            np.zeros(shape=[Dim]), requires_grad=True
        )  # Output is multi-variate

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    def discriminator(new_x, h):
        inputs = torch.cat(dim=1, tensors=[new_x, h])  # Hint + Data Concatenate
        D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
        D_logit = torch.matmul(D_h2, D_W3) + D_b3
        D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
        return D_prob

    # 2. Generator
    if torch.cuda.is_available():
        G_W1 = torch.tensor(
            xavier_init([Dim * 2, H_Dim1]), requires_grad=True, device="cuda"
        )  # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = torch.tensor(np.zeros(shape=[H_Dim1]), requires_grad=True, device="cuda")

        G_W2 = torch.tensor(
            xavier_init([H_Dim1, H_Dim2]), requires_grad=True, device="cuda"
        )
        G_b2 = torch.tensor(np.zeros(shape=[H_Dim2]), requires_grad=True, device="cuda")

        G_W3 = torch.tensor(
            xavier_init([H_Dim2, Dim]), requires_grad=True, device="cuda"
        )
        G_b3 = torch.tensor(np.zeros(shape=[Dim]), requires_grad=True, device="cuda")
    else:
        G_W1 = torch.tensor(
            xavier_init([Dim * 2, H_Dim1]), requires_grad=True
        )  # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = torch.tensor(np.zeros(shape=[H_Dim1]), requires_grad=True)

        G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]), requires_grad=True)
        G_b2 = torch.tensor(np.zeros(shape=[H_Dim2]), requires_grad=True)

        G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]), requires_grad=True)
        G_b3 = torch.tensor(np.zeros(shape=[Dim]), requires_grad=True)

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    def generator(new_x, m):
        inputs = torch.cat(dim=1, tensors=[new_x, m])  # Mask + Data Concatenate
        G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)
        G_prob = torch.sigmoid(
            torch.matmul(G_h2, G_W3) + G_b3
        )  # [0,1] normalized Output
        return G_prob

    # 3. Other functions
    # Random sample generator for Z
    def sample_Z(m, n):
        return np.random.uniform(0.0, 0.01, size=[m, n])

    # Mini-batch generation
    def sample_idx(m, n):
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    # Hint vector generation
    def sample_M(m, n, p):
        A = np.random.uniform(0.0, 1.0, size=[m, n])
        B = A > p
        C = 1.0 * B
        return C

    def discriminator_loss(M, New_X, H):
        G_sample = generator(New_X, M)
        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = discriminator(Hat_New_X, H)

        # Loss
        D_loss = -torch.mean(
            M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1.0 - D_prob + 1e-8)
        )
        return D_loss

    def generator_loss(X, M, New_X, H):
        G_sample = generator(New_X, M)
        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = discriminator(Hat_New_X, H)

        # Loss
        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
        MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)
        G_loss = G_loss1 + alpha * MSE_train_loss

        # MSE Performance metric
        MSE_test_loss = torch.mean(
            ((1 - M) * X - (1 - M) * G_sample) ** 2
        ) / torch.mean(1 - M)
        return G_loss, MSE_train_loss, MSE_test_loss

    def test_loss(X, M, New_X):
        G_sample = generator(New_X, M)

        # MSE Performance metric
        MSE_test_loss = torch.mean(
            ((1 - M) * X - (1 - M) * G_sample) ** 2
        ) / torch.mean(1 - M)
        return MSE_test_loss, G_sample

    optimizer_D = torch.optim.Adam(params=theta_D)
    optimizer_G = torch.optim.Adam(params=theta_G)

    MSE = []

    while len(MSE) < repetition:
        rep = len(MSE) + 1
        D_loss_last = 0
        # Start Iterations
        for it in range(iteration):

            # Inputs
            mb_idx = sample_idx(Train_No, batch_size)
            X_mb = trainX[mb_idx, :]

            Z_mb = sample_Z(batch_size, Dim)
            M_mb = trainM[mb_idx, :]

            if is_dh:
                H_mb1 = sample_M(
                    batch_size,
                    Dim,
                    1 - 1 / (1 + math.e ** (-(10 * D_loss_last - 5)))
                    if it > 0
                    else 1 - hr,
                )
            else:
                H_mb1 = sample_M(batch_size, Dim, 1 - hr)

            if is_fi:
                H_mb = M_mb * H_mb1 + (1 - M_mb) * (1 - H_mb1)
                New_X_mb = M_mb * X_mb + Z_mb
            else:
                H_mb = M_mb * H_mb1
                New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

            if torch.cuda.is_available():
                X_mb = torch.tensor(X_mb, device="cuda")
                M_mb = torch.tensor(M_mb, device="cuda")
                H_mb = torch.tensor(H_mb, device="cuda")
                New_X_mb = torch.tensor(New_X_mb, device="cuda")
            else:
                X_mb = torch.tensor(X_mb)
                M_mb = torch.tensor(M_mb)
                H_mb = torch.tensor(H_mb)
                New_X_mb = torch.tensor(New_X_mb)

            optimizer_D.zero_grad()
            D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
            D_loss_curr.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(
                X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb
            )
            G_loss_curr.backward()
            optimizer_G.step()

            # Intermediate Losses
            if it % 100 == 0:
                print("Repetition: {}".format(rep), end="\t")
                print("Iteration: {}".format(it), end="\t")
                print(
                    "Train_loss: {:.4}".format(np.sqrt(MSE_train_loss_curr.item())),
                    end="\t",
                )
                print("Test_loss: {:.4}".format(np.sqrt(MSE_test_loss_curr.item())))

            D_loss_last = D_loss_curr.item()

        Z_mb = sample_Z(Test_No, Dim)
        M_mb = testM
        X_mb = testX

        if is_fi:
            New_X_mb = M_mb * X_mb + Z_mb
        else:
            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        if torch.cuda.is_available():
            X_mb = torch.tensor(X_mb, device="cuda")
            M_mb = torch.tensor(M_mb, device="cuda")
            New_X_mb = torch.tensor(New_X_mb, device="cuda")
        else:
            X_mb = torch.tensor(X_mb)
            M_mb = torch.tensor(M_mb)
            New_X_mb = torch.tensor(New_X_mb)

        MSE_final, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)
        MSE.append(np.sqrt(MSE_final.item()))

    print("\nRMSE list: " + str(MSE))
    print("\nRMSE mean: " + str(np.mean(MSE)))
    print("\nRMSE standard deviation: " + str(np.std(MSE)))
    print()
