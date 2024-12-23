#!/usr/local/bin/python3
import os
import time
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datafolder.folder import Train_Dataset, Train_Custom_Dataset
from net import get_model

def main():
    # Settings
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    dataset_dict = {
        'market': 'Market-1501',
        'duke': 'DukeMTMC-reID',
        'custom': 'Custom-dataset'
    }

    # Argument
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data-path', default='./classification_singletask/dataset', type=str, help='path to the dataset')
    parser.add_argument('--dataset', default='custom', type=str, help='dataset: market, duke, custom')
    parser.add_argument('--backbone', default='resnet50', type=str, help='backbone: resnet50, resnet34, resnet18, densenet121')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
    parser.add_argument('--num-workers', default=0, type=int, help='num_workers')
    args = parser.parse_args()

    assert args.dataset in ['market', 'duke', 'custom']
    assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

    dataset_name = dataset_dict[args.dataset]
    model_name = '{}_nfc'.format(args.backbone)
    data_dir = args.data_path
    model_dir = os.path.join('./classification_singletask/checkpoints', args.dataset, model_name)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Save model function
    def save_network(network, epoch_label):
        save_filename = 'net_%s.pth' % epoch_label
        save_path = os.path.join(model_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if use_gpu:
            network.cuda()
        print('Save model to {}'.format(save_path))

    # Draw curve
    x_epoch = []
    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []

    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig(os.path.join(model_dir, 'train.jpg'))

    # DataLoader
    image_datasets = {}
    if args.dataset == 'custom':
        image_datasets['train'] = Train_Custom_Dataset(data_dir)
        image_datasets['val'] = Train_Custom_Dataset(data_dir, train_val='val')
    else:
        image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_name, train_val='train')
        image_datasets['val'] = Train_Dataset(data_dir, dataset_name=dataset_name, train_val='query')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
                  for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    num_label = image_datasets['train'].num_label()
    labels_list = image_datasets['train'].labels()

    # Model and Optimizer
    model = get_model(model_name, num_label)
    if use_gpu:
        model = model.cuda()

    # loss
    criterion_bce = nn.BCELoss()

    # optimizer
    ignored_params = list(map(id, model.features.parameters()))
    classifier_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = torch.optim.SGD([
                {'params': model.features.parameters(), 'lr': 0.01},
                {'params': classifier_params, 'lr': 0.1},
            ], momentum=0.9, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # Training function
    def train_model(model, optimizer, scheduler, num_epochs):
        since = time.time()

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train(True)  # Set model to training mode
                else:
                    model = model.cpu()  # Sposta il modello sulla CPU per la validazione
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for count, (images, indices, labels, ids, cams, names) in enumerate(dataloaders[phase]):
                    labels = labels.float()

                    if phase == 'train' and use_gpu:
                        images = images.cuda()
                        labels = labels.cuda()

                    # Crea una maschera per escludere le etichette -1
                    valid_mask = labels != -1  # La maschera è True dove le etichette sono 0 o 1

                    # Applica la maschera
                    pred_label = model(images)
                    masked_pred_label = pred_label[valid_mask]  # Output del modello solo per etichette valide
                    masked_labels = labels[valid_mask]  # Etichette valide

                    # Calcola la perdita solo per le etichette valide
                    if masked_pred_label.numel() > 0:  # Verifica che ci siano elementi validi
                        total_loss = criterion_bce(masked_pred_label, masked_labels)

                        if phase == 'train':
                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()
                            scheduler.step()  # Correct order
                            torch.cuda.empty_cache()  # Libera memoria inutilizzata dalla GPU

                        preds = torch.gt(pred_label, torch.ones_like(pred_label) / 2)
                        running_loss += total_loss.item()
                        running_corrects += torch.sum(preds == labels.byte()).item() / num_label
                        if count % 100 == 0:
                            print('step: ({}/{})  |  label loss: {:.4f}'.format(
                                count * args.batch_size, dataset_sizes[phase], total_loss.item()))

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0 - epoch_acc)

                if phase == 'val':
                    model = model.cuda() if use_gpu else model  # Riporta il modello su GPU (se disponibile) dopo la validazione
                    last_model_wts = model.state_dict()
                    if epoch % 10 == 0:
                        save_network(model, epoch)
                    draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        model.load_state_dict(last_model_wts)
        save_network(model, 'last')

    # Main training loop
    train_model(model, optimizer, exp_lr_scheduler, num_epochs=args.num_epoch)

if __name__ == '__main__':
    main()
