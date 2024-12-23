import os
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model


######################################################################
# Settings
# ---------
if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
    'custom' : 'Custom-dataset'
}
num_cls_dict = { 'market':30, 'duke':23 ,'custom':3}

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='Path to test image')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
args = parser.parse_args()

assert args.dataset in ['market', 'duke', 'custom']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

model_name ='{}_nfc'.format(args.backbone)
num_label= num_cls_dict[args.dataset]


######################################################################
# Model and Data
# ---------
def load_network(network):
    save_path = os.path.join('./classification_singletask/checkpoints', args.dataset, model_name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path, weights_only=True))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src


model = get_model(model_name, num_label)
model = load_network(model)
model.eval()

src = load_image(args.image_path)

######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open('./classification_singletask/doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./classification_singletask/doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))


out = model.forward(src)

pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5

Dec = predict_decoder(args.dataset)
Dec.decode(pred)

