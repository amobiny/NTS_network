import warnings
warnings.filterwarnings("ignore")
import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, DATA, PROPOSAL_NUM, SAVE_FREQ, LR, WD, DISP_FREQ, resume, save_dir
from core import model
from utils.logger_utils import Logger
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if DATA == 'CUB':
    from core.dataset import CUB as data
    data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/NTS_network/CUB_200_2011'
elif DATA == 'ISIC':
    from core.skin_dataset import ISIC as data
    data_dir = ''
elif DATA == 'cheXpert':
    from core.chexpert_dataset import CheXpertDataSet as data
    data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/CheXpert-v1.0-small'
else:
    raise NameError('Dataset not available!')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(save_dir)

model_dir = os.path.join(save_dir, 'models')
os.makedirs(model_dir)

logs_dir = os.path.join(save_dir, 'tf_logs')

# bkp of config def
os.system('cp {}/config.py {}'.format(BASE_DIR, save_dir))
# bkp of model def
os.system('cp {}/core/model.py {}'.format(BASE_DIR, save_dir))
# bkp of train procedure
os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
if DATA == 'cheXpert':
    os.system('cp {}/core/chexpert_dataset.py {}'.format(BASE_DIR, save_dir))


LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(save_dir) + '\n')

train_logger = Logger(os.path.join(logs_dir, 'train'))
test_logger = Logger(os.path.join(logs_dir, 'val'))

# read dataset
trainset = data(root=data_dir, is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)
testset = data(root=data_dir, is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)
# define model
net = model.attention_net(topN=PROPOSAL_NUM)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)

global_step = 0
best_test_loss = 100.
best_test_acc = 0
num_batches = len(trainset) // BATCH_SIZE

for epoch in range(start_epoch, 500):
    log_string('*********** EPOCH %03d ***********' % epoch)
    log_string('---- TRAINING')
    sys.stdout.flush()
    train_loss = 0
    train_correct = 0
    total = 0

    for scheduler in schedulers:
        scheduler.step()

    # begin training
    net.train()
    for batch_idx, data in enumerate(trainloader):
        global_step += 1
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()

        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
        part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        raw_loss = creterion(raw_logits, label)
        concat_loss = creterion(concat_logits, label)
        rank_loss = model.ranking_loss(top_n_prob, part_loss)
        partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()

        # calculate loss
        concat_loss = creterion(concat_logits, label)
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)
        total += batch_size
        train_correct += torch.sum(concat_predict.data == label.data)
        train_loss += concat_loss.item() * batch_size

        if batch_idx % DISP_FREQ == 0:
            print('CURRENT_BATCH/TOTAL: %d/%d' % (batch_idx, num_batches))

    if epoch % SAVE_FREQ == 0:
        # train_loss = 0
        # train_correct = 0
        # total = 0
        # net.eval()
        # for batch_idx, data in enumerate(trainloader):
        #     with torch.no_grad():
        #         img, label = data[0].cuda(), data[1].cuda()
        #         batch_size = img.size(0)
        #         _, concat_logits, _, _, _ = net(img)
        #         # calculate loss
        #         concat_loss = creterion(concat_logits, label)
        #         # calculate accuracy
        #         _, concat_predict = torch.max(concat_logits, 1)
        #         total += batch_size
        #         train_correct += torch.sum(concat_predict.data == label.data)
        #         train_loss += concat_loss.item() * batch_size
        #     if batch_idx % DISP_FREQ == 0:
        #         print('CURRENT_BATCH/TOTAL: %d/%d' % (batch_idx, num_batches))

        train_acc = float(train_correct) / total
        train_loss = train_loss / total
        info = {'loss': train_loss, 'accuracy': train_acc}
        for tag, value in info.items():
            train_logger.scalar_summary(tag, value, global_step)
        log_string('train loss: {:.3f}, train acc: {:.01%}'.format(
                train_loss,
                train_acc))

        # evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        log_string('---- VALIDATION')
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size

        test_acc = float(test_correct) / total
        test_loss = test_loss / total

        info = {'loss': test_loss, 'accuracy': test_acc}
        for tag, value in info.items():
            test_logger.scalar_summary(tag, value, global_step)

        text_ = ''
        if test_loss <= best_test_loss and test_acc >= best_test_acc:
            text_ = '(*** improved ***)'
            best_test_loss = test_loss
            best_test_acc = test_acc
        elif test_loss <= best_test_loss:
            text_ = '(*** loss improved ***)'
            best_test_loss = test_loss
        elif test_acc >= best_test_acc:
            text_ = '(*** acc improved ***)'
            best_test_acc = test_acc
        log_string(
            'test loss: {:.3f}, test acc: {:.01%} {}'.format(
                test_loss,
                test_acc,
                text_))

    # save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(model_dir, '%03d.ckpt' % epoch)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'net_state_dict': net_state_dict},
            save_path)
        log_string("Model saved in file: %s" % save_path)

print('finishing training')
