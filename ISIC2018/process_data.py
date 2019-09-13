import h5py
import numpy as np
from sklearn.model_selection import train_test_split

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Skin Lesion/BNN_Skin_Lesion/data/ISIC/data.h5', 'r')
X = h5f['X'][:]
y = h5f['y'][:]
h5f.close()

num_samples = y.shape[0]
img_height = X.shape[1]
img_width = X.shape[2]

# split train and test data
print('Start splitting the data ...')
train_idx = np.array([])
y_train = np.array([])
test_idx = np.array([])
y_test = np.array([])
for cls in range(7):
    print('class #{}'.format(cls))
    # get the data for each class
    cls_idx = np.where(y == cls)[0]
    y_cls = cls * np.ones(cls_idx.shape[0])
    # split and concatenate
    tr_idx, te_idx, y_tr, y_te = train_test_split(cls_idx, y_cls, test_size=0.2, random_state=42)
    train_idx = np.append(train_idx, tr_idx)
    test_idx = np.append(test_idx, te_idx)
    y_train = np.append(y_train, y_tr)
    y_test = np.append(y_test, y_te)

print('Done.')

h5f = h5py.File('ISIC_data.h5', 'w')
h5f.create_dataset('train_idx', data=train_idx)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('test_idx', data=test_idx)
h5f.create_dataset('y_test', data=y_test)
h5f.close()
