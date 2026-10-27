import torch
import numpy as np
import random
import os
import pickle
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
# 导入我们的模块
from config import Config
from dataset import PrivateDataset, PublicDataset, public_collate_fn, private_collate_fn
from trainer import Trainer
from model import ECAN
from torch.utils.data import random_split
# import matplotlib.pyplot as plt

# 设置随机种子
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(best_model_dir):
    # 调整模式
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")


    print(">>>  Evaluating Best Model... <<<")
    DATA_PATH= Config.get_datapath(Config.DATASET_NMAE)
    public_dataset = PublicDataset(DATA_PATH)
    
    n_splits = Config.n_splits

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)

    all_acc = []
    all_f1 = []


    model_path = os.listdir(best_model_dir)

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(public_dataset)))):
        
        # 根据索引创建子集
        train_subset = Subset(public_dataset, train_idx)
        val_subset = Subset(public_dataset, val_idx)

        # 注意: Public 数据集需要特定的 collate_fn 来处理变长序列 padding
        train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, 
                                shuffle=True, collate_fn=public_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, 
                                shuffle=False, collate_fn=public_collate_fn)
        print(f"--- FOLD {fold+1} ---", f'Val Set: {len(val_subset)}')
    
        trainer = Trainer(device)

        # 确保加载 Phase 2 训练好的 Student 权重
        student_ckpt = os.path.join(Config.SAVE_DIR, "student_phase2_best.pth")
        # 组装 H-EDAPT
        classify_model = ECAN(pretrained_student_path=student_ckpt).to(device)
        
        best_model = [item for item in model_path if f'fold{fold+1}' in item][0]
        
        print('验证阶段加载的最优模型路径为', best_model)
        classify_model.load_state_dict(torch.load(os.path.join(best_model_dir, best_model)))
        classify_model.eval()


        # 开始验证模型效果
        all_preds = []
        all_labels = []
            
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio_feats'].to(device)
                text = batch['text_feats'].to(device)
                stim = batch['stim_embeds'].to(device)
                mask = batch['mask'].to(device)
                labels = batch['label'].to(device)
                output = classify_model(audio, text, stim, mask)
                logits = output['logits_cls']
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_acc.append(accuracy_score(all_labels, all_preds))
        all_f1.append(f1_score(all_labels, all_preds))

        # 计算最终指标
        print(classification_report(all_labels, all_preds, digits=4))
    print(f'Average Acc:{np.mean(all_acc)}, Average F1:{np.mean(all_f1)}')



# ==========================================
# 主流程
# ==========================================
def main():
    # 1. 环境设置
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
   
    # 初始化 Trainer
    trainer = Trainer(device)
    
    # ==========================================
    # Phase 1: Teacher Warm-up
    # ==========================================
    print("\n" + "="*40)
    print(" PHASE 1: Training Teacher (VAE)")
    print("="*40)
    
    # 加载 Private 数据
    private_dataset_train =  PrivateDataset(Config.PRIVATE_DATA_PATH_TRAIN)
    private_dataset_test =  PrivateDataset(Config.PRIVATE_DATA_PATH_TEST)
    train_private_loader = DataLoader(private_dataset_train, batch_size=Config.BATCH_SIZE, collate_fn=private_collate_fn, shuffle=True)
    val_private_loader = DataLoader(private_dataset_test, batch_size=Config.BATCH_SIZE, collate_fn=private_collate_fn)
    print('train_private: ', len(private_dataset_train), 'test_private: ', len(private_dataset_test))
    # 开始训练 Phase 1
    # 如果已有权重不想重训，可以注释掉这行
    trainer.train_phase1(train_private_loader, val_private_loader, epochs=Config.EPOCHS_WARMUP)
    
    # ==========================================
    # Phase 2: Knowledge Dis tillation
    # ==========================================
    print("\n" + "="*40)
    print(" PHASE 2: Distilling to Student")
    print("="*40)
    
    # Phase 2 复用 Private 数据，但 Dataset 逻辑一样 (return_raw=True 已经在 dataset.py 里是默认行为)
    # 确保加载了刚才训练好的 Teacher 权重
    teacher_ckpt = os.path.join(Config.SAVE_DIR, "teacher_phase1_best.pth")
    
    # 开始训练 Phase 2
    trainer.train_phase2(train_private_loader, val_private_loader, teacher_path=teacher_ckpt, epochs=Config.EPOCHS_STUDENT)
    
    # ==========================================
    # Phase 3: Joint Training (Depression Detection)
    # ==========================================
    print("\n" + "="*40)
    print(" PHASE 3: Joint Training (Aggregator)")
    print("="*40)
    
    # 加载 Public 数据 (层级结构)
    # 构建 Dataset
    if Config.DATASET_NMAE == 'daic':
        fold = 0
        train_dataset = PublicDataset(Config.PUBLIC_DATA_PATH,split='train')
        test_dataset = PublicDataset(Config.PUBLIC_DATA_PATH,split='test')

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                                    shuffle=True, collate_fn=public_collate_fn)
        val_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                                    shuffle=False, collate_fn=public_collate_fn)
        
        print(f'Train Set: {len(train_dataset)}, Val Set: {len(test_dataset)}')
            
        # 确保加载 Phase 2 训练好的 Student 权重
        student_ckpt = os.path.join(Config.SAVE_DIR, "student_phase2_best.pth")
        
        # 开始训练 Phase 3
        best_acc, best_f1, cm = trainer.train_phase3(Config.DATASET_NMAE, train_loader, val_loader, student_path=student_ckpt, epochs=Config.EPOCHS_JOINT, FOLD=0, conf_save=Config.SAVE_DIR)
        
        print(f'Best Acc: {best_acc}, Best F1: {best_f1}')

        print("\n" + "="*40)
        print(" ALL TRAINING COMPLETED SUCCESSFULLY!")
        print(f" Checkpoints saved in {Config.SAVE_DIR}")
        print("="*40)
        
    
    else:
        public_dataset = PublicDataset(Config.PUBLIC_DATA_PATH)

        n_splits = Config.n_splits

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)

        all_acc = []
        all_f1 = []

    
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(public_dataset)))):
            
            # 根据索引创建子集
            train_subset = Subset(public_dataset, train_idx)
            val_subset = Subset(public_dataset, val_idx)

            # 注意: Public 数据集需要特定的 collate_fn 来处理变长序列 padding
            train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, 
                                    shuffle=True, collate_fn=public_collate_fn)
            val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, 
                                    shuffle=False, collate_fn=public_collate_fn)
            print(f"--- FOLD {fold+1} ---", f'Train Set: {len(train_subset)}, Val Set: {len(val_subset)}')
            
            # 确保加载 Phase 2 训练好的 Student 权重
            student_ckpt = os.path.join(Config.SAVE_DIR, "student_phase2_best.pth")
            
            # 开始训练 Phase 3
            best_acc, best_f1, cm = trainer.train_phase3(Config.DATASET_NMAE, train_loader, val_loader, student_path=student_ckpt, epochs=Config.EPOCHS_JOINT, FOLD=fold+1, conf_save=Config.SAVE_DIR)
            all_acc.append(best_acc)
            all_f1.append(best_f1)

            print("\n" + "="*40)
            print(" ALL TRAINING COMPLETED SUCCESSFULLY!")
            print(f" Checkpoints saved in {Config.SAVE_DIR}")
            print("="*40)

        print(f'Average Acc:{np.mean(all_acc)}, Average F1:{np.mean(all_f1)}')

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    Config.AGG_LAYER_PRINT=False
    if Config.DATASET_NMAE == 'modma':
        Config.CurrentStimus = 18
    elif Config.DATASET_NMAE == 'midd':
        Config.CurrentStimus = 9 
    elif Config.DATASET_NMAE == 'daic':
        Config.CurrentStimus = 14
        # Config.Max_patience = 20
    elif Config.DATASET_NMAE == 'cmdc':
        Config.CurrentStimus = 12
    
    # Training model:
    main()   
    
    # if you need to evluate our model 
    
    # best_model_dir = '/home/zhanghb/MyPaperProgram/MyPaper/EmoGeneralization/code_v14/checkpoints/modma/best'
    # best_model_dir = '/home/zhanghb/MyPaperProgram/MyPaper/EmoGeneralization/code_v13/checkpoints/midd/best'
    # best_model_dir = '/home/zhanghb/MyPaperProgram/MyPaper/EmoGeneralization/code_v14/checkpoints/cmdc/best'
    # evaluate_model(best_model_dir)