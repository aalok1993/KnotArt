import cv2
import torch

def preprocess_image(img_path, img_size, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    img = 1 - img
    img = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_AREA)
    I = torch.tensor(img).to(device)
    return I

def get_geodesic_dist(csum, start, end):
    length = (csum[end] - csum[start]).abs()
    return torch.minimum(length, csum[-1] - length)

def get_euclidean_dist(points, start, end):
    length = (points[end] - points[start]).norm(dim=-1)
    return length

def chamfer_loss(X,Y):
    dist_sq = (X.unsqueeze(1) - Y.unsqueeze(0)).square().sum(-1)
    return dist_sq.min(-1)[0].mean() + dist_sq.min(-2)[0].mean()