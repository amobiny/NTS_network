import os
from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
from config import BATCH_SIZE, PROPOSAL_NUM, test_model, DATA, data_dir, DISP_FREQ
from core import model, dataset
from core.py_utils import progress_bar

if DATA == 'CUB':
    from core.dataset import CUB as data
elif DATA == 'ISIC':
    from core.skin_dataset import ISIC as data
elif DATA == 'cheXpert':
    from core.chexpert_dataset import CheXpertDataSet as data
else:
    raise NameError('Dataset not available!')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


save_dir = os.path.dirname(os.path.dirname(test_model))
LOG_FOUT = open(os.path.join(save_dir, 'log_test.txt'), 'w')
LOG_FOUT.write(str(save_dir) + '\n')
viz_dir = os.path.join(save_dir, 'visualize')
train_viz = os.path.join(viz_dir, 'train')
test_viz = os.path.join(viz_dir, 'test')
if not os.path.exists(viz_dir):
    os.mkdir(viz_dir)
    os.mkdir(train_viz)
    os.mkdir(test_viz)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if not test_model:
    raise NameError('please set the test_model file to choose the checkpoint!')

# read dataset
trainset = data(root=data_dir, is_train=True, data_len=100)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=8, drop_last=False)
testset = data(root=data_dir, is_train=False, data_len=100)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)

# define model
net = model.attention_net(topN=PROPOSAL_NUM)
log_string('Loading {}'.format(test_model))
ckpt = torch.load(test_model)
net.load_state_dict(ckpt['net_state_dict'])
net = net.cuda()
net = DataParallel(net)
creterion = torch.nn.CrossEntropyLoss()
log_string('Model Successfully loaded!')

# evaluate on train set
train_loss = 0
train_correct = 0
total = 0
net.eval()

log_string('Evaluating on the Training set')
num_batches = len(trainset) // BATCH_SIZE
for i, data in enumerate(trainloader):
    with torch.no_grad():
        img, label, indeces = data[0].cuda(), data[1].cuda(), data[2].cpu().numpy()
        batch_size = img.size(0)
        _, concat_logits, _, _, _ = net(img, indeces, img_save_path=train_viz)
        # calculate loss
        concat_loss = creterion(concat_logits, label)
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)
        total += batch_size
        train_correct += torch.sum(concat_predict.data == label.data)
        train_loss += concat_loss.item() * batch_size
        # progress_bar(i, len(trainloader), 'eval on train set')
        if i % DISP_FREQ == 0:
            print('CURRENT_BATCH/TOTAL: %d/%d' % (i, num_batches))

train_acc = float(train_correct) / total
train_loss = train_loss / total
log_string('train set loss: {:.3f} and train set acc: {:.01%}'.format(train_loss, train_acc))


# evaluate on test set
log_string('Evaluating on the Training set')
num_batches = len(testset) // BATCH_SIZE
test_loss = 0
test_correct = 0
total = 0
for i, data in enumerate(testloader):
    with torch.no_grad():
        img, label, indeces = data[0].cuda(), data[1].cuda(), data[2].cpu().numpy()
        batch_size = img.size(0)
        _, concat_logits, _, _, _ = net(img, indeces, img_save_path=test_viz)
        # calculate loss
        concat_loss = creterion(concat_logits, label)
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)
        total += batch_size
        test_correct += torch.sum(concat_predict.data == label.data)
        test_loss += concat_loss.item() * batch_size
        if i % DISP_FREQ == 0:
            print('CURRENT_BATCH/TOTAL: %d/%d' % (i, num_batches))

test_acc = float(test_correct) / total
test_loss = test_loss / total
log_string('test set loss: {:.3f} and test set acc: {:.01%}'.format(test_loss, test_acc))

print('finishing testing')
