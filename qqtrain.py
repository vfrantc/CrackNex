from dataset.fewshot import FewShot
from model.QQCrackNex_matching import CrackNex
from util.utils import count_params, set_seed, calc_crack_pixel_weight, mIOU

import argparse
from copy import deepcopy
import os
import time
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Mining Latent Classes for Few-shot Segmentation')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        default='./Datasets_CrackNex/LCSD',
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='LCSD',
                        choices=['llCrackSeg9k', 'LCSD'],
                        help='training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--loss',
                        type=str,
                        choices=['CE', 'weightedCE'],
                        default='CE',
                        help='loss function')
    parser.add_argument('--crop-size',
                        type=int,
                        default=400,
                        help='cropping size of training samples')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet101',
                        help='backbone of semantic segmentation model')

    # few-shot training arguments
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--episode',
                        type=int,
                        default=6000,
                        choices=[6000, 18000, 24000, 36000],
                        help='total episodes of training')
    parser.add_argument('--snapshot',
                        type=int,
                        default=200,
                        choices=[200, 1200, 2000],
                        help='save the model after each snapshot episodes')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')

    args = parser.parse_args()
    return args

def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    num_classes = 3

    metric = mIOU(num_classes)
    for i, (img_s_list, hiseq_s_list, mask_s_list, img_q, hiseq_q, mask_q, cls, _, id_q) in enumerate(tbar):
        img_q, hiseq_q, mask_q = img_q.cuda(), hiseq_q.cuda(), mask_q.cuda()
        for k in range(len(img_s_list)):
            img_s_list[k], hiseq_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), hiseq_s_list[k].cuda(), mask_s_list[k].cuda()
        cls = cls[0].item()

        with torch.no_grad():
            out_ls = model(img_s_list, hiseq_s_list, mask_s_list, img_q, hiseq_q, mask_q)
            pred = torch.argmax(out_ls[0], dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():
    args = parse_args()
    print('\n' + str(args))

    save_path = 'outdir/models/%s' % (args.dataset)
    os.makedirs(save_path, exist_ok=True)

    trainset = FewShot(args.data_root, args.crop_size,
                       'train', args.shot, args.snapshot)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=0, drop_last=True)
    testset = FewShot(args.data_root, None, 'val',
                    args.shot, 41 if args.dataset == 'LCSD' else 1486)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=0, drop_last=False)

    model = CrackNex(args.backbone)
    print('\nParams: %.1fM' % count_params(model))

    #for param in model.rgb_layer0.parameters():
    #    param.requires_grad = False
    #for param in model.rgb_layer1.parameters():
    #    param.requires_grad = False
    # for param in model.ref_layer0.parameters():
    #     param.requires_grad = False
    # for param in model.ref_layer1.parameters():
    #     param.requires_grad = False

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    if args.loss == 'CE':
        criterion = CrossEntropyLoss(ignore_index=255)
    elif args.loss == 'weightedCE':
        crack_weight = [1, 0.4] * calc_crack_pixel_weight(args.data_root)
        print(f'positive weight: {crack_weight}')
        criterion = CrossEntropyLoss(weight=torch.Tensor([crack_weight]).to('cuda').squeeze(0), ignore_index=255)

    optimizer = SGD([param for param in model.parameters() if param.requires_grad],
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    model = DataParallel(model).cuda()
    best_model = None

    iters = 0
    total_iters = args.episode // args.batch_size
    lr_decay_iters = [total_iters // 3, total_iters * 2 // 3]
    previous_best = 0

    # each snapshot is considered as an epoch
    for epoch in range(args.episode // args.snapshot):
        print("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        tbar = tqdm(trainloader)
        set_seed(int(time.time()))

        for i, (img_s_list, hiseq_s_list, mask_s_list, img_q, hiseq_q, mask_q, _, _, _) in enumerate(tbar):
            img_q, hiseq_q, mask_q = img_q.cuda(), hiseq_q.cuda(), mask_q.cuda()
            for k in range(len(img_s_list)):
                img_s_list[k], hiseq_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), hiseq_s_list[k].cuda(), mask_s_list[k].cuda()

            out_ls = model(img_s_list, hiseq_s_list, mask_s_list, img_q, hiseq_q, mask_q)
            mask_s = torch.cat(mask_s_list, dim=0)

            loss = criterion(out_ls[0], mask_q) + criterion(out_ls[1], mask_q) + criterion(out_ls[2], mask_q) + criterion(out_ls[3], mask_s) * 0.2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            if iters in lr_decay_iters:
                optimizer.param_groups[0]['lr'] /= 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        model.eval()
        set_seed(args.seed)
        miou = evaluate(model, testloader, args)

        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

    torch.save(best_model.module.state_dict(),
               os.path.join(save_path, '%s_%ishot_%.2f.pth' % (args.backbone, args.shot, total_miou / 5)))


if __name__ == '__main__':
    main()
