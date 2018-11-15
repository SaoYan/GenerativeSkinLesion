import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Generator, Discriminator

if __name__ == "__main__":
    D = Discriminator()
    print(D)
    print('\n\n\n\n')

    for i in range(2,7):
        D.grow_network()
        D.flush_network()

    D.grow_network()
    print(D)
    print('\n\n\n\n')
    D.flush_network()
    print(D)
    print('\n\n\n\n')
