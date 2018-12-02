import os
import cv2
import numpy as np
import torch
from networks import Generator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

def normalize_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)

if __name__ == "__main__":
    # define model
    G = Generator(nc=3, nz=512, size=256)
    for i in range(6):
        G.grow_network()
        G.flush_network()

    # load checkpoint
    checkpoint = torch.load('stage7.pth')
    G.load_state_dict(checkpoint['G_EMA_state_dict'])
    G.eval()
    G.to(device)

    # generate and save images
    with torch.no_grad():
        for num in range(1250):
            z = torch.FloatTensor(1, 512).normal_(0.0, 1.0).to(device)
            generate_data = G.forward(z)
            image = generate_data.to('cpu').squeeze()
            # image = image.clamp(-1., 1.).add(1.).div(2.)
            image = normalize_tensor(image)
            image = np.transpose(image.numpy(), (1,2,0))
            image *= 255
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            filename = os.path.join('Images_Gen', 'ISIC_gen_{:07d}.jpg'.format(num))
            print(filename)
            cv2.imwrite(filename, image)
