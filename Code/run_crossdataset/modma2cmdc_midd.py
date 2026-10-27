import sys
import os 
import torch 
import numpy as np 
import itertools
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

from trainer import Trainer
from run_cross import run, set_seed
from config import Config
from dataset import PublicDataset, public_collate_fn
from model import H_EDAPT


def evaluate_model(best_model_path, val_dataset_name):
    # 调整模式
    Cross_stimu = False
    Cross_Data = True


    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")


    print(">>>  Evaluating Best Model... <<<")
    DATA_PATH= Config.get_datapath(val_dataset_name, CrossStimu=Cross_stimu, CrossData=Cross_Data)
    val_dataset = PublicDataset(DATA_PATH)
    
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, collate_fn=public_collate_fn)
    print(f'Val Set: {len(val_dataset)}')

    trainer = Trainer(device)

    # 确保加载 Phase 2 训练好的 Student 权重
    student_ckpt = os.path.join(Config.SAVE_DIR, "student_phase2_best.pth")
    # 组装 H-EDAPT
    classify_model = H_EDAPT(pretrained_student_path=student_ckpt).to(device)
    print('验证阶段加载的最优模型路径为', best_model_path)
    classify_model.load_state_dict(torch.load(best_model_path))
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
    
    # 计算最终指标
    print(classification_report(all_labels, all_preds, digits=4))
    

if __name__ == '__main__':
    Config.AGG_LAYER_PRINT=False
    Config.CurrentStimus=9
    acc, f1 = run(dataset_name1='modma', dataset_name2='cmdc_midd', LR=1e-3, epochs=50, CrossData=True)




    


