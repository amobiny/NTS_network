BATCH_SIZE = 16
NUM_CLS = 2
    # for CUB: 200
    # for cheXpert: 14
PROPOSAL_NUM = 6
CAT_NUM = 4
INPUT_SIZE = (448, 448)  # (w, h)
LR = 0.001
WD = 1e-4
SAVE_FREQ = 1
DISP_FREQ = 100
resume = ''
DATA = 'cheXpert'    # 'CUB' or 'ISIC' or 'cheXpert'
test_model = 'model.ckpt'
save_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/NTS_network/save/'
data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/CheXpert-v1.0-small'
    # for CUB: '/home/cougarnet.uh.edu/amobiny/Desktop/NTS_network/CUB_200_2011'
    # for cheXpert: '/home/cougarnet.uh.edu/amobiny/Desktop/CheXpert-v1.0-small'

