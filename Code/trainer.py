import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
from tqdm import tqdm
import numpy as np
import os
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
# import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import copy

# 导入所有定义好的模块
from config import Config
from losses import LossFactory
from teacher import TeacherNetwork
from student import StudentEncoder
from model import ECAN
# from utils.visualizer import AggregateVisualizer 


# def setup_logger(save_dir):
#     # 1. 创建日志目录
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # 2. 获取 logger 实例
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
    
#     # 防止重复添加 handler（如果你在脚本中多次调用 setup_logger）
#     if not logger.handlers:
#         # 3. 创建一个 Formatter，定义日志格式
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

#         # 4. 创建 FileHandler，用于写入文件
#         log_file = os.path.join(save_dir, "train_log.txt")
#         file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)

#         # 5. 创建 StreamHandler，用于输出到控制台
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(formatter)
#         logger.addHandler(console_handler)
    
#     return logger


# 配置简单的日志打印格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """
    H-EDAPT 全流程训练管理器。
    负责 Phase 1 (Teacher), Phase 2 (Distillation), Phase 3 (Classification) 的调度。
    """
    def __init__(self, device):
        self.device = device
        self.config = Config
        self.loss_factory = LossFactory(Config)
        
        # 路径准备
        if not os.path.exists(Config.SAVE_DIR):
            os.makedirs(Config.SAVE_DIR)
            
        # 模型占位符 (按需初始化，节省显存)
        self.teacher = None
        self.student = None
        self.ecan = None

    # =================================================================
    # Phase 1: Teacher Warm-up 
    # =================================================================
    def _get_teacher_forcing_ratio(self, epoch, total_epochs):
        
        k = 20.0 # 衰减速度控制
        ratio = k / (k + np.exp(epoch / k))
        
        # 或者使用更简单的线性退火
        start_ratio = 1.0
        end_ratio = 0.0
        decay_epochs = 20 # 在前20个epoch内衰减完
        
        if epoch < decay_epochs:
            ratio = start_ratio - (start_ratio - end_ratio) * (epoch / decay_epochs)
        else:
            ratio = 0.0
            
        return max(0.0, ratio)
    

    def train_phase1(self, train_loader, test_loader, epochs=50):
        logger.info(">>> Starting Phase 1: Teacher Warm-up <<<")

        # 1. 初始化模型
        self.teacher = TeacherNetwork().to(self.device)
        optimizer = optim.Adam(self.teacher.parameters(), lr=Config.LR_TEACHER)
        # 这里的 scheduler 改为监控 val_loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf') # 用于记录最好的验证集 Loss
        
        for epoch in range(epochs):
            # ==========================
            #       Training Loop
            # ==========================
            self.teacher.train() # 开启 Teacher Forcing (如果有设置) 和 Dropout
            total_loss = 0
            total_recon = 0
            total_kld = 0
            # total_corr = 0
            # total_cls = 0
            total_contra = 0
            
            for batch in train_loader:
                # Flattened Data: (B*N, ...)
                stim_embeds = batch['stim_embeds'].to(self.device)
                emo_seqs = batch['emo_seqs'].to(self.device)
                labels = batch['labels'].to(self.device) # private dataset labels
                
                optimizer.zero_grad()
                
                # Forward
                ratio = self._get_teacher_forcing_ratio(epoch, epochs)

                # train 模式下，Decoder可能会使用 Teacher Forcing (取决于 teacher.py 实现)
                recon_emo, (post_mu, post_logvar), (prior_mu, prior_logvar), aux_class_logits = self.teacher(stim_embeds, emo_seqs, ratio)
                
                # Loss Calculation
                loss_dict = self.loss_factory.teacher_loss(
                    recon_emo, emo_seqs, 
                    (post_mu, post_logvar), (prior_mu, prior_logvar), 
                    aux_class_logits, labels,
                    epoch
                )
                loss = loss_dict['loss']
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=5.0)
                optimizer.step()
                
                # Logging Stats
                total_loss += loss.item()
                total_recon += loss_dict['recon_loss'] # 确保转为 float
                total_kld += loss_dict['kld_loss']
                # total_corr += loss_dict['corr_loss']
                # total_cls += loss_dict['cls_loss']
                total_contra += loss_dict['contra_loss']

            avg_train_loss = total_loss / len(train_loader)

            # ==========================
            #      Validation Loop
            # ==========================
            self.teacher.eval() # 关闭 Teacher Forcing -> 自回归生成测试！
            val_loss = 0
            val_steps = 0

                       
            # 用于后续可能需要绘图的数据容器 
            val_real, val_recon, val_labels, val_stims = [], [], [], []
            
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    stim_embeds = batch['stim_embeds'].to(self.device)
                    emo_seqs = batch['emo_seqs'].to(self.device)
                    labels = batch['labels'].to(self.device)     # 必须有 label 用于绘图验证
                    stim_ids = batch['stim_ids'].to(self.device) # 必须有 stim_id 用于绘图验证
                    
                    # Forward (Eval 模式)
                    # 注意：虽然这里传入了 emo_seqs，但 eval() 模式下 Decoder 应该使用自回归生成上一帧
                    # 传入 emo_seqs 主要是为了让 Encoder 计算 Posterior (重构能力测试)
                    recon_emo, (post_mu, post_logvar), (prior_mu, prior_logvar), aux_class_logits = self.teacher(stim_embeds, emo_seqs, ratio)
                    
                    # Validation Loss (衡量泛化能力)
                    loss_dict = self.loss_factory.teacher_loss(
                        recon_emo, emo_seqs, 
                        (post_mu, post_logvar), (prior_mu, prior_logvar), 
                        aux_class_logits, labels,
                        epoch
                    )
                    val_loss += loss_dict['loss'].item()
                    val_steps += 1
                    
                    # --- 收集全量数据用于绘图  ---
                    val_real.append(emo_seqs.cpu().numpy())
                    val_recon.append(recon_emo.cpu().numpy())
                    val_labels.append(labels.cpu().numpy())
                    val_stims.append(stim_ids.cpu().numpy())

            avg_val_loss = val_loss / val_steps

            # --- 收集全量数据用于绘图  ---
            all_real = np.concatenate(val_real, axis=0)   # (Total_Val_Samples, 50, 7)
            all_recon = np.concatenate(val_recon, axis=0) # (Total_Val_Samples, 50, 7)
            all_labels = np.concatenate(val_labels, axis=0)
            all_stim_ids = np.concatenate(val_stims, axis=0)
            
            # ==========================
            #    Logging & Scheduling
            # ==========================
            scheduler.step(avg_val_loss) # 根据验证集 Loss 调整学习率
            
            logger.info(f"Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {avg_train_loss:.4f} (Recon: {total_recon/len(train_loader):.4f}, Kld: {total_kld/len(train_loader):.4f}, con: {total_contra/len(train_loader):.4f}) | "
                        f"Val Loss: {avg_val_loss:.4f}")

            # ==========================
            #      Model Saving
            # ==========================
            # 1. 保存当前 Epoch 模型 (覆盖)
            torch.save(self.teacher.state_dict(), os.path.join(Config.SAVE_DIR, "teacher_phase1_latest.pth"))
            
            # 2. 保存历史最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(Config.SAVE_DIR, "teacher_phase1_best.pth")
                torch.save(self.teacher.state_dict(), save_path)
                logger.info(f"*** 🎉 New Best Model Saved (Val Loss: {best_val_loss:.4f}) ***")
                
        logger.info(f"Phase 1 Completed. Best Val Loss: {best_val_loss:.4f}")

    # =================================================================
    # Phase 2: Knowledge Distillation (Student Training)
    # =================================================================
    def train_phase2(self, train_loader, val_loader, teacher_path, epochs=50):
        logger.info(">>> Starting Phase 2: Knowledge Distillation <<<")
        
        # 1. 加载 Teacher (冻结)
        if self.teacher is None:
            self.teacher = TeacherNetwork().to(self.device)
        self.teacher.load_state_dict(torch.load(teacher_path))
        self.teacher.eval() 
        for param in self.teacher.parameters():
            param.requires_grad = False # 冻结 Teacher 的所有梯度，只当做“评委”
            
        # 2. 初始化 Student
        self.student = StudentEncoder().to(self.device)
        optimizer = optim.Adam(self.student.parameters(), lr=Config.LR_STUDENT)
        
        # 最佳模型监控
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # ==========================================
            # Training Loop
            # ==========================================
            self.student.train()
            train_loss = 0
            train_align = 0
            train_sem = 0
            train_contra = 0
            
            # 进度条可选，为了日志清晰这里用 logger
            for batch in train_loader:
                # Private Dataset 必须包含 Audio/Text
                audio_feats = batch['audio_feats'].to(self.device) # (K, 1, D)
                text_feats = batch['text_feats'].to(self.device)   # (K, 1, D)
                stim_embeds = batch['stim_embeds'].to(self.device) # (K, 768)
                emo_seqs = batch['emo_seqs'].to(self.device)       # (K, T, 7) - Teacher 的参考答案
                labels = batch['labels'].to(self.device) 
                
                optimizer.zero_grad()
                
                # 1. Teacher Generate Targets (No Grad)
                # Teacher 看着“答案”(emo_seqs) 生成“真理”(Posterior Z)
                with torch.no_grad():
                    _, (t_post_mu, t_post_logvar), _, _ = self.teacher(stim_embeds, emo_seqs)
                
                # 2. Student Predicts (Grad)
                # Student 看着“题目”(Audio/Text) 猜测 Z 分布
                # 注意：Student 必须尽量去拟合 Teacher 的 Posterior，而不是 Prior
                s_z, s_mu, s_logvar, _ = self.student(audio_feats, text_feats, stim_embeds)
                
                # 3. Semantic Reconstruction (通过 Teacher Decoder)
                # 让 Student 的 Z 通过 Teacher 的 Decoder 画图
                # teacher_forcing_ratio=0.0: 强迫 Student 生成完整序列，不依赖 GT，这样 Loss 才能反映 Student 的真实水平
                s_recon_emo = self.teacher.decode(z=s_z, stim_embeds=stim_embeds, emo_seqs=None, teacher_forcing_ratio=0.0)

                aux_class_logits = self.teacher.aux_classifier(s_mu)
                #  print(s_mu.shape, aux_class_logits.shape, labels.shape)
                # 4. Loss Calculation
                loss_dict = self.loss_factory.distillation_loss(
                    student_mu=s_mu, student_logvar=s_logvar,
                    teacher_mu=t_post_mu, teacher_logvar=t_post_logvar, # 目标是 Teacher Posterior
                    target_labels=labels,
                    student_recon_emo=s_recon_emo, target_emo=emo_seqs,
                    aux_class_logits=aux_class_logits
                )
                
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_align += loss_dict.get('align_loss', 0) # 对应 keys 视 losses.py 而定
                train_sem += loss_dict.get('sem_loss', 0)
                train_contra += loss_dict.get('contra', 0)
            
            avg_train_sem = train_sem / len(train_loader)
            avg_train_loss = train_loss / len(train_loader)
            avg_train_align = train_align / len(train_loader)

            # Validation Loop 
            self.student.eval()
            val_loss = 0
            val_sem_loss = 0
            # val_cls_loss = 0
            val_align_loss = 0
            
            
            with torch.no_grad():
                for batch in val_loader:
                    audio_feats = batch['audio_feats'].to(self.device)
                    text_feats = batch['text_feats'].to(self.device)
                    stim_embeds = batch['stim_embeds'].to(self.device)
                    emo_seqs = batch['emo_seqs'].to(self.device)
                    labels = batch['labels'].to(self.device) 

                    # 1. Teacher Targets
                    _, (t_post_mu, t_post_logvar), _, _ = self.teacher(stim_embeds, emo_seqs)

                    # 2. Student Prediction
                    s_z, s_mu, s_logvar, _ = self.student(audio_feats, text_feats, stim_embeds)
                    
                    if torch.isnan(s_z).any():
                        print("IN model!!! CRITICAL: NaN detected in Student Output (s_z) !!!")
                    
                    if torch.isnan(s_mu).any():
                        print("IN model!!! CRITICAL: NaN detected in Student Output (s_mu) !!!")
                        
                    if torch.isnan(s_logvar).any():
                        print("IN model!!! CRITICAL: NaN detected in Student Output (s_logvar) !!!")

                    # 3. Decode
                    s_recon_emo = self.teacher.decode(z=s_z, stim_embeds=stim_embeds, emo_seqs=None, teacher_forcing_ratio=0.0)
                    aux_class_logits = self.teacher.aux_classifier(s_mu)
                    # 4. Loss
                    loss_dict = self.loss_factory.distillation_loss(
                        student_mu=s_mu, student_logvar=s_logvar,
                        teacher_mu=t_post_mu, teacher_logvar=t_post_logvar,
                        target_labels=labels,
                        student_recon_emo=s_recon_emo, target_emo=emo_seqs,
                        aux_class_logits=aux_class_logits
                    )
                    
                    val_loss += loss_dict['loss'].item()
                    val_sem_loss += loss_dict.get('sem_loss', 0)
                    # val_cls_loss += loss_dict.get('cls_loss', 0)
                    val_align_loss += loss_dict['align_loss']

            avg_val_loss = val_loss / len(val_loader)
            avg_val_sem = val_sem_loss / len(val_loader)
            # avg_val_cls = val_cls_loss / len(val_loader)  
            avg_val_align = val_align_loss / len(val_loader)          

            logger.info(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} (Sem: {avg_train_sem:.4f} align: {avg_train_align:.4f}) | Val Loss: {avg_val_loss:.4f} (Sem: {avg_val_sem:.4f}, align: {avg_val_align:.4f})")

            # Save Best Model

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(Config.SAVE_DIR, "student_phase2_best.pth")
                torch.save(self.student.state_dict(), save_path)
                logger.info(f"    >>> 🎉 Best Student Saved! (Val Loss: {best_val_loss:.4f})")

        # 训练结束，保存最后一个 Epoch
        last_path = os.path.join(Config.SAVE_DIR, "student_phase2_last.pth")
        torch.save(self.student.state_dict(), last_path)
        logger.info(f"Phase 2 Completed.")

    

    # =================================================================
    # Phase 3: Joint Training & Classification
    # =================================================================
    def train_phase3(self, datasetname, train_loader, val_loader, student_path, epochs, FOLD, conf_save=Config.SAVE_DIR, LR=None, Cross=False,
                     use_teacher=True, freeze_student=True, noise_z=False):
        """
        Phase 3: Aggregator 联合训练。
        
        消融实验开关：
            use_teacher:    是否在 ECAN 中加载 Teacher（False 跳过 teacher.decode）。
            freeze_student: 是否冻结 Student 参数（False 时 Student 端到端训练，使用全 LR_STUDENT）。
            noise_z:        是否将送入 Aggregator 的 Z_student 替换为 N(0,1) 噪声（验证情绪表征有效性）。
        """
        logger.info(">>> Starting Phase 3: Depression Detection (Joint Training) <<<")
        
        # 组装 H-EDAPT
        self.ecan = ECAN(pretrained_student_path=student_path, use_teacher=use_teacher,
                         freeze_student=freeze_student, noise_z=noise_z).to(self.device)
        
        if LR is None:
            base_lr_agg = Config.LR_AGGREGATOR[Config.DATASET_NMAE]
        else:
            base_lr_agg = LR

        # print('LR:', LR)

        # Student LR：消融端到端训练用全 LR，原模式用 /10（实际不更新）
        student_lr = Config.LR_STUDENT if not freeze_student else Config.LR_STUDENT / 10
        optimizer = optim.AdamW([ 
            {'params': self.ecan.student.parameters(), 'lr': student_lr }, 
            {'params': self.ecan.aggregator.parameters(), 'lr': base_lr_agg} 
        ], weight_decay=0.05) # 增加微弱权重衰减提高稳定性

        if Config.Lr_Scheduler:
            warmup_scheduler = LinearLR(optimizer, start_factor=Config.Start_f, end_factor=Config.End_f, total_iters=Config.Total_iters)
            cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, T_mult=Config.T_mult)
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=Config.Milestones)
        

        best_val_acc = 0.0
        patience = 0
        max_patience = Config.Max_patience # 早停耐心值

        if Config.DETAILA_SAVEA_NAME:
            best_name = f"best_model_seed_{Config.SEED}_lr_{base_lr_agg}_agg_feed_{Config.AGG_FEED_FORWARD}_layer_{Config.AGG_TSF_LAYERS}_nhead_{Config.AGG_N_HEAD}_drop_{Config.AGG_DROPOUT}_align_{Config.COEFFI_ALIGN}_epoch{epochs}"
        else:
            best_name = 'best_model'
        
        final_metrics = {
            'best_acc': 0.0,
            'f1_score': 0.0,
            'confusion_matrix': None}
        
        best_model_path = None

        for epoch in range(epochs):
            # --- Training ---
            self.ecan.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            
            # 获取当前学习率以便记录
            curr_lr_stu = optimizer.param_groups[0]['lr']
            curr_lr_agg = optimizer.param_groups[1]['lr']
            # print('curr_lr_agg,', curr_lr_agg)
            # print('curr_lr_stu,', curr_lr_stu)
            
            for batch in train_loader:
                # Public Dataset: Hierarchical Data
                audio = batch['audio_feats'].to(self.device) # (B, N, 1, D)
                text = batch['text_feats'].to(self.device)
                stim = batch['stim_embeds'].to(self.device)
                mask = batch['mask'].to(self.device)
                # print('train: ', mask.shape, mask[0])
                labels = batch['label'].to(self.device)      # (B,)
                optimizer.zero_grad()
                
                # Forward (End-to-End)
                output_dict = self.ecan(audio, text, stim, mask)
                logits = output_dict['logits_cls']  
                # Loss
                loss_dict = self.loss_factory.aggregator_loss(output_dict, labels)
                loss = loss_dict['loss']
                
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ecan.parameters(), max_norm=1.0) 
                optimizer.step()
                
                total_loss += loss.item()
                
                # print('loss item:', loss.item())
                # Metrics
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                # pbar.set_postfix({'Loss': loss.item(), 'Acc': train_correct/train_total})
                # print(optimizer.param_groups[0]['lr'])
            
            # --- 更新学习率 ---
            if Config.Lr_Scheduler:
                scheduler.step()
            train_acc = train_correct / train_total
            
            # --- Validation ---
            val_loss, val_acc = self.evaluate_phase3(val_loader)
            
            if Config.AGG_LAYER_PRINT:
                logger.info(f"Fold {FOLD} Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")
            
            # --- Early Stopping & Checkpointing ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                if Cross:
                    save_path = os.path.join(conf_save, f"{datasetname}_{best_name}_acc{best_val_acc:.4f}.pth")
                else:
                    save_path = os.path.join(conf_save, Config.DATASET_NMAE ,f"{datasetname}_{best_name}_fold{FOLD}_acc{best_val_acc:.4f}.pth")
                best_model_copy = copy.deepcopy(self.ecan)
                best_model_path = save_path
                logger.info(f"🎉 New Best Model! Acc: {best_val_acc:.4f}, Epoch {epoch+1}")
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info("Early stopping triggered.")
                    torch.save(best_model_copy.state_dict(), best_model_path)
                    break   
            # save            
            if epoch == epochs - 1 :
                torch.save(best_model_copy.state_dict(), best_model_path)
                
           

        # --- Final Evaluation using Best Model ---
        logger.info(">>> Training Complete. Evaluating Best Model... <<<")
        # 加载最优权重
        print('验证阶段加载的最优模型路径为', best_model_path)
        self.ecan.load_state_dict(torch.load(best_model_path))
        self.ecan.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio_feats'].to(self.device)
                text = batch['text_feats'].to(self.device)
                stim = batch['stim_embeds'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)
                output_dict = self.ecan(audio, text, stim, mask)
                logits = output_dict['logits_cls']
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算最终指标
        final_metrics['best_acc'] = accuracy_score(all_labels, all_preds)
        final_metrics['f1_score'] = f1_score(all_labels, all_preds, average='binary')
        final_metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
        
        logger.info(f"Final Best Acc: {final_metrics['best_acc']:.4f}")
        logger.info(f"Final Binary F1: {final_metrics['f1_score']:.4f}")
        print(classification_report(all_labels, all_preds, digits=4))

        
        return final_metrics['best_acc'], final_metrics['f1_score'], final_metrics['confusion_matrix']
        

    def evaluate_phase3(self, val_loader):
        self.ecan.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio_feats'].to(self.device)
                text = batch['text_feats'].to(self.device)
                stim = batch['stim_embeds'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)
                # print('test: ', mask.shape, mask[0])
                output_dict = self.ecan(audio, text, stim, mask)
                loss_dict = self.loss_factory.aggregator_loss(output_dict, labels)

                logits = output_dict['logits_cls']  
                
                total_loss += loss_dict['loss'].item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return total_loss / len(val_loader), correct / total
