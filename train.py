# -------------------------------------#
#       对自己的数据集进行训练，获取模型
# -------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.yolov4 import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from util.callbacks import LossHistory
from util.dataloader import YoloDataset, yolo_dataset_collate
from util.utils import get_anchors, get_classes, get_lr


def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda):
    if Tensorboard:
        global train_tensorboard_step, val_tensorboard_step
    loss = 0
    val_loss = 0

    model_train.train()
    print('\nStart Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(images)

            loss_value_all = 0  # =0
            num_pos_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for l in range(len(outputs)):
                loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
                num_pos_all += num_pos
            loss_value = loss_value_all / num_pos_all  # 无sum

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()

            # if Tensorboard:
            #     # 将loss写入tensorboard，每一步都写
            #     writer = SummaryWriter(log_dir='logs', flush_secs=60)
            #     writer.add_scalar('Train_loss', loss, train_tensorboard_step)
            #     train_tensorboard_step += 1

            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # 将loss写入tensorboard，下面是每个世代保存一次
    if Tensorboard:
        # writer = SummaryWriter(log_dir='logs', flush_secs=60)
        writer.add_scalar('Train_loss', loss / (iteration + 1), epoch)

    model_train.eval()
    print('Finish Train')
    print('\nStart Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                # ----------------------#
                #   清零梯度
                # ----------------------#
                optimizer.zero_grad()
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(images)

                loss_value_all = 0  # =0
                num_pos_all = 0
                # ----------------------#
                #   计算损失
                # ----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                    num_pos_all += num_pos
                loss_value = loss_value_all / num_pos_all
                val_loss += loss_value.item()

            # 将val_loss写入tensorboard, 下面注释的是每一步都写入
            # if Tensorboard:
            #     writer = SummaryWriter(log_dir='logs', flush_secs=60)
            #     writer.add_scalar('Val_loss', loss, val_tensorboard_step)
            #     val_tensorboard_step += 1
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    # 将loss写入tensorboard，每个世代保存一次
    if Tensorboard:
        # writer = SummaryWriter(log_dir='logs', flush_secs=60)  #查看tensorboard的端口号
        writer.add_scalar('Val_loss', val_loss / (epoch_step_val + 1), epoch)

    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if epoch == 0:
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(),
                   'logs/Epoch%d-loss%.4f-val_loss%.4f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
    elif (epoch + 1) % 10 == 0:
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(),
                   'logs/Epoch%d-loss%.4f-val_loss%.4f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))


if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Tensorboard
    # -------------------------------#
    Tensorboard = True
    # -------------------------------#
    #   是否使用Cuda 确认GPU是否满足
    # -------------------------------#
    Cuda = True
    # --------------------------------------------------------#
    #   训练前classes_path，使其对应自己的数据集路径里面为目标类别class
    # --------------------------------------------------------#
    classes_path = 'model_data/xray_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，下面的pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # ----------------------------------------------------------------------------------------------------------------------------#
    #训练时请修改预训练权重模型
    model_path = 'model_data/LightRay.pth'
    # ------------------------------------------------------#
    #   输入图片的shape大小，一定要是32的倍数
    # ------------------------------------------------------#
    input_shape = [416, 416]
    # -------------------------------#
    #  主干特征提取网络mobilenetv3
    # -------------------------------#
    backbone = "mobilenetv3"

    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    pretrained = False
    # ------------------------------------------------------#
    #   Yolov4的一些tricks应用，本实验针对数据集没有使用
    #   mosaic 马赛克数据增强 True or False
    #   测试时mosaic数据增强并不稳定，所以为False
    #   余弦退火学习率 Cosine_scheduler = True or False
    #   标签平滑  label_smoothing  = 0.01一般为0.01、0.005
    # ------------------------------------------------------#
    mosaic = False
    Cosine_lr = False
    label_smoothing = 0.005

    Init_Epoch = 0
    Freeze_Epoch = 100
    Freeze_batch_size = 8
    Freeze_lr = 1e-3

    UnFreeze_Epoch = 200
    Unfreeze_batch_size = 4
    Unfreeze_lr = 1e-4
    # ------------------------------------------------------#
    #   是否进行冻结训练
    # ------------------------------------------------------#
    Freeze_Train = True

    num_workers = 8
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = '2021_train.txt'
    val_annotation_path = '2021_val.txt'

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    # ------------------------------------------------------#
    #   创建模型
    # ------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':

        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        # model_train = torch.nn.DataParallel(model)
        model_train = model
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory("logs/")

    # ---------------------------#
    #   读取数据集，格式为txt
    # ---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    # --------------------查看模型----------------#
    if Tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(log_dir='logs', flush_secs=5)
        if Cuda:
            graph_inputs = torch.randn(1, 3, input_shape[0], input_shape[1]).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.randn(1, 3, input_shape[0], input_shape[1]).type(torch.FloatTensor)
        writer.add_graph(model, graph_inputs)
        train_tensorboard_step = 1
        val_tensorboard_step = 1

    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ------------------------------------#
        #   冻结训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ------------------------------------#
        #   冻结训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
