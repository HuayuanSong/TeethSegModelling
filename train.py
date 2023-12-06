import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mesh_dataset import *
from dl_models import *
from loss_func import *
import utils

if __name__ == '__main__':
    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
    use_visdom = True # if you don't use visdom, please set to False

    train_list = './train_list.csv' # use 1-fold as example
    val_list = './val_list.csv' # use 1-fold as example

    model_path = 'models/'
    model_name = 'model' # need to define

    num_classes = 15
    num_channels = 15 #number of features
    num_epochs = 200
    num_workers = 0
    train_batch_size = 10
    val_batch_size = 10
    num_batches_to_print = 20

    if use_visdom:
        # set plotter
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=model_name)

    # mkdir 'models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # set dataset
    training_dataset = Mesh_Dataset(data_list_path=train_list,
                                    num_classes=num_classes,
                                    patch_size=6000)
    val_dataset = Mesh_Dataset(data_list_path=val_list,
                               num_classes=num_classes,
                               patch_size=6000)

    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Option: Change to PointNet
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    opt = optim.Adam(model.parameters(), amsgrad=True)

    losses = []
    val_losses = []

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Training model...')
    class_weights = torch.ones(15).to(device, dtype=torch.float)
    for epoch in range(num_epochs):

        # training
        model.train()
        running_loss = 0.0
        loss_epoch = 0.0
        for i_batch, batched_sample in enumerate(train_loader):

            # send mini-batch to device
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)
            A_S = batched_sample['A_S'].to(device, dtype=torch.float)
            A_L = batched_sample['A_L'].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, A_S, A_L)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            loss_epoch += loss.item()
            if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print))
                if use_visdom:
                    plotter.plot('loss', 'train', 'Loss', epoch+(i_batch+1)/len(train_loader), running_loss/num_batches_to_print)
                running_loss = 0.0

        # record losses and metrics
        losses.append(loss_epoch/len(train_loader))

        #reset
        loss_epoch = 0.0

        # validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            val_loss_epoch = 0.0

            for i_batch, batched_val_sample in enumerate(val_loader):

                # send mini-batch to device
                inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                labels = batched_val_sample['labels'].to(device, dtype=torch.long)
                A_S = batched_val_sample['A_S'].to(device, dtype=torch.float)
                A_L = batched_val_sample['A_L'].to(device, dtype=torch.float)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                outputs = model(inputs, A_S, A_L)
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)

                running_val_loss += loss.item()
                val_loss_epoch += loss.item()

                if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                    print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print))
                    running_val_loss = 0.0

            # record losses and metrics
            val_losses.append(val_loss_epoch/len(val_loader))

            # reset
            val_loss_epoch = 0.0

            # output current status
            print('*****\nEpoch: {}/{}, loss: {}\n         val_loss: {}\n*****'.format(epoch+1, num_epochs, losses[-1], val_losses[-1]))
            if use_visdom:
                plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
                plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])

        # save the checkpoint
        #torch.save({'epoch': epoch+1,
                    #'model_state_dict': model.state_dict(),
                    #'optimizer_state_dict': opt.state_dict(),
                    #'losses': losses,
                    #'val_losses': val_losses},
                    #model_path+"checkpoint1")

        # save the best model
        if best_val_losses < val_losses[-1]:
            best_val_losses = val_losses[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'val_losses': val_losses},
                        model_path+'{}_best.tar'.format(model_name))
