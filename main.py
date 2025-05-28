import os
import numpy as np
import argparse
from data.data_reader import *
from data.augmentation import *
from loss.loss import *
from train.train import *
from utils.utils import *
from model.simclr_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

def main():
    parser = argparse.ArgumentParser(description="Train a SimCLR on Landslide dataset")
    parser.add_argument("--patch_size", type = int, default = 6, help="half size of patch")
    parser.add_argument("--pre_batch_size", type = int, default = 128, help = "batch size of the pretraining model")
    parser.add_argument("--batch_size", type = int, default = 32, help = "batch size of fine tuning model")
    parser.add_argument("--pre_epochs", type = int, default = 15, help = "epochs of pre training model")
    parser.add_argument("--strides", type = int, default = 6, help = "strides when make patch from image")

    parser.add_argument("--pre_learning_rate", type = float, default = 0.0001, help = "learning rate of pretraining model")
    parser.add_argument("--learning_rate", type = float, default = 0.00001, help = "learning rate of fine tuning model")
    parser.add_argument("--fine_tuning_data_ratio", type = float, default = 1e-1, help = "ratio of dataset used on fine tuing")

    parser.add_argument("--height", type = list, default = [554270.0, 562860.0, 859], help = "information of height")
    parser.add_argument("--width", type = list, default = [331270.0, 342070.0, 1080], help = "information of width")   

    parser.add_argument("--random_aug", type = bool, default = False, help = "Whether to apply random augmentation or not")

    parser.add_argument("--aug_1", type = str, default = 'flip', help = "Whether to have flip augmentation")
    parser.add_argument("--aug_2", type = str, default = 'flip', help = "Whether to have rotation augmentation")
    parser.add_argument("--tif_img_path", type = str, default = './data/tif_img.npy', help = "tif image path")
    parser.add_argument("--ls_img_path", type = str, default = './data/ls_img.npy', help = "landslide image path")
    parser.add_argument("--pre_trained_model_name", type = str, default = "pre", help = "name of the pre-trained model")
    parser.add_argument("--fine_trained_model_name", type = str, default = "fine", help = "name of the fine-trained model")
    parser.add_argument("--dir_name", type = str, default = "250228", help = "directory name of fine-tuned model to save")
    parser.add_argument("--pre_model", type = str, default = "SimCLR", help = "['SimCLR', 'CNN', 'VIF'] sort of pre-training model")
    
    args = parser.parse_args()

    tif_img = np.load(args.tif_img_path)
    ls_img = np.load(args.ls_img_path)
    ls_img = np.expand_dims(ls_img, 0)
    tif_img = np.concatenate([tif_img, ls_img], axis=0)

    real_patches, non_ls_patches_features, ls_patches_features, non_ls_patches_labels, ls_patches_labels, coor_list = patch_(args.height, args.width, tif_img, args.patch_size, args.strides)

    multi_data_scale = Multi_data_scaler(real_patches)
    train_scaled = multi_data_scale.multi_scale(real_patches)

    train_scaled = np.resize(train_scaled, (train_scaled.shape[0], 28, 28, 20))

    np.random.seed(110)
    np.random.shuffle(train_scaled)

    data_1 = []
    data_2 = []
    for i in range(len(train_scaled)):
        view1 = apply_simclr_augmentation(train_scaled[i], args.aug_1, args.random_aug)
        view2 = apply_simclr_augmentation(train_scaled[i], args.aug_2, args.random_aug)
        view1 = np.expand_dims(view1, axis=0)
        view2 = np.expand_dims(view2, axis=0)
        data_1.append(view1)
        data_2.append(view2)

    data_1 = np.concatenate(data_1)
    data_2 = np.concatenate(data_2)

    data_1 =  np.resize(data_1, (data_1.shape[0], 28, 28, 20))
    data_2 =  np.resize(data_2, (data_2.shape[0], 28, 28, 20))

    
    # 폴더 안의 모든 항목을 확인
    if args.fine_tuning_data_ratio<2e-2:  
        with tf.device('/CPU:0'):
            pretrain_model = pretrain(data_1, data_2, batch_size=args.pre_batch_size, pre_model=args.pre_model, epochs=args.pre_epochs)
            pretrain_model.save("./pretrained_model/%s.h5"%args.pre_trained_model_name)
    else:
        if args.pre_model == 'SimCLR':
            pretrain_model = tf.keras.models.load_model('./pretrined_model/random_aug/random_aug_1.h5', custom_objects={'residual_block': residual_block})
        elif args.pre_model == 'CNN':
            pretrain_model = tf.keras.models.load_model('./pretrined_model/random_aug_cnn/random_aug_1.h5')
        elif args.pre_model == 'VIF':
            pretrain_model = tf.keras.models.load_model('./pretrined_model/random_aug_vif/random_aug_1.h5', custom_objects={'PositionalEmbedding': PositionalEmbedding})

    ls_patches_features = np.load("./data/landslides.npy")
    non_ls_patches_features = np.load("./data/non_landslides.npy")
    ls_patches_labels = np.load("./data/landslides_labels.npy")
    non_ls_patches_labels = np.load("./data/non_landslides_labels.npy")

    train_non_ls_ind = np.random.choice(non_ls_patches_features.shape[0], int(ls_patches_labels.shape[0]*2*args.fine_tuning_data_ratio))
    train_ls_ind = np.random.choice(ls_patches_features.shape[0], int(ls_patches_features.shape[0]*args.fine_tuning_data_ratio))

    train_non_ls_features_0 = non_ls_patches_features[train_non_ls_ind]
    train_non_ls_labels_0 = non_ls_patches_labels[train_non_ls_ind]

    ls_patches_features = ls_patches_features[train_ls_ind]
    ls_patches_labels = ls_patches_labels[train_ls_ind]    

    train_ls, valid_ls, train_ls_label, valid_ls_label = train_test_split(ls_patches_features,ls_patches_labels, test_size=0.3, random_state=42)
    train_non_ls, valid_non_ls, train_non_ls_label, valid_non_ls_label = train_test_split(train_non_ls_features_0, train_non_ls_labels_0, test_size=0.3, random_state=42)

    train_patches = np.concatenate([train_ls, train_non_ls])
    valid_patches = np.concatenate([valid_ls, valid_non_ls])
    train_labels = np.concatenate([train_ls_label, train_non_ls_label])
    valid_labels = np.concatenate([valid_ls_label, valid_non_ls_label])

    train_labels_ = []
    valid_labels_ = []
    for i in range(train_labels.shape[0]):
        if np.sum(train_labels[i, args.patch_size-1:args.patch_size+1, args.patch_size-1:args.patch_size+1]) >=1.:
            ls_bool = 1
            train_labels_.append(ls_bool)
        else:
            ls_bool = 0
            train_labels_.append(ls_bool)

    for i in range(valid_labels.shape[0]):
        if np.sum(valid_labels[i, args.patch_size-1:args.patch_size+1, args.patch_size-1:args.patch_size+1]) >=1.:
            ls_bool = 1
            valid_labels_.append(ls_bool)
        else:
            ls_bool = 0
            valid_labels_.append(ls_bool)

    train_labels_ = np.expand_dims(train_labels_, axis=1)
    valid_labels_ = np.expand_dims(valid_labels_, axis=1)

    train_patches = np.resize(train_patches, (train_patches.shape[0], 28, 28, 20))
    valid_patches = np.resize(valid_patches, (valid_patches.shape[0], 28, 28, 20))

    train_patches_scaled = multi_data_scale.multi_scale(train_patches)
    valid_patches_scaled = multi_data_scale.multi_scale(valid_patches)

    input_shape = train_patches_scaled.shape[1:]

    encoder = pretrain_model.get_layer("encoder")

    finetune_model = build_finetune_model(input_shape, encoder, num_classes=2, training=False)
    
    finetune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])
    try:
        os.makedirs('./finetuned_models/%s'%args.dir_name)
    except:
        print("Directory name %s"%args.dir_name, 'exist')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("./finetuned_models/%s/%s.h5"%(args.dir_name, args.fine_trained_model_name), save_best_only=True)
    checkpoint_cb2 = tf.keras.callbacks.ModelCheckpoint("./finetuned_models/%s/%s_weight.h5"%(args.dir_name, args.fine_trained_model_name), save_weight_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20,
                                                    restore_best_weights=True)
    reduce_lr_plateu_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

    with tf.device('/CPU:0'):
        history_2 = finetune_model.fit(train_patches_scaled, train_labels_,
                        validation_data = (valid_patches_scaled, valid_labels_),
                        epochs=200,
                        batch_size=32,
                        callbacks = [checkpoint_cb,
                                checkpoint_cb2,
                                early_stopping_cb,
                                reduce_lr_plateu_cb,
                                    ]
                                    )
    try:
        os.makedirs('./loss_curve/%s'%args.dir_name)
    except:
        pass
    plt.plot(history_2.history['val_loss'], label = 'valid_loss')
    plt.plot(history_2.history['loss'], label = 'train_loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('Categorical_Cross_Entropy')
    plt.title('Loss curve')
    plt.tight_layout()
    plt.savefig('./loss_curve/%s/%s'%(args.dir_name,args.fine_trained_model_name))

    non_ls_patches_scaled = multi_data_scale.multi_scale(non_ls_patches_features)
    with tf.device('/CPU:0'):
        full_result = finetune_model.predict(non_ls_patches_scaled)

    coor_list = np.load('./data/non_ls_coordinate.npy')

    plt.rc('font', size=15) 

    cmap = ListedColormap(['darkgreen', 'lawngreen', 'palegoldenrod','coral','firebrick'])
    variable_list = []
    caution = ['Very Low','Low','Moderate','High','Very High']
    for i in range(5):
        variable = mpatches.Patch(color = cmap.colors[i], label='%s'%caution[i])
        variable_list.append(variable)

    variable_list.append(Line2D([0], [0], marker='o', color='w', label = 'Landslide', markerfacecolor='k', markersize=3))

    try:
        os.makedirs('./images/landslide_maps/%s'%args.dir_name)
    except:
        pass

    ls_coor = np.load('./data/ls_coor.npy')
    ls_label = np.load('./data/ls_labels.npy')

    plt.figure(figsize=(10, 6))
    plt.scatter(coor_list[:,5,5,0], coor_list[:,5,5,1], s= 0.1, c = full_result[:,1], cmap= "RdYlGn_r")
    plt.scatter(ls_coor[:,0], ls_coor[:,1], s=3, c=ls_label[:,0], cmap='gray')
    plt.legend(handles =variable_list, bbox_to_anchor=(1.02, 1))

    plt.title('%s susceptibility'%args.fine_trained_model_name)
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.tight_layout()
    plt.savefig('./images/landslide_maps/%s/%s.png'%(args.dir_name, args.fine_trained_model_name))

    test_patches = np.load("./data/test_patches.npy")
    test_ls = np.load("./data/test_labels.npy")

    test_patches_scaled = multi_data_scale.multi_scale(test_patches)

    with tf.device('/CPU:0'):
        test_results = finetune_model.predict(test_patches_scaled)

    test_label = test_ls[:,5,5]
    test_label[test_label==0.001] = 0
    
    fine_pred = tf.where(test_results >= 0.5, 1, 0)
    kp_score = cohen_kappa_score(test_label, fine_pred[:,1])
    f1 = f1_score(test_label, fine_pred[:,1])
    acc = accuracy_score(test_label, fine_pred[:,1])
    best_fp_clu, best_tp_clu, _ = roc_curve(test_label,test_results[:,1:])
    auc_score = auc(best_fp_clu, best_tp_clu)

    try:
        os.makedirs('./performance/%s'%args.dir_name)
    except:
        pass

    np.save('./performance/%s/%s_acc.npy'%(args.dir_name, args.fine_trained_model_name), acc)
    np.save('./performance/%s/%s_kappa.npy'%(args.dir_name, args.fine_trained_model_name), kp_score)
    np.save('./performance/%s/%s_f1.npy'%(args.dir_name, args.fine_trained_model_name), f1)
    np.save('./performance/%s/%s_auc.npy'%(args.dir_name, args.fine_trained_model_name), auc_score)

if __name__ == "__main__":
    main()