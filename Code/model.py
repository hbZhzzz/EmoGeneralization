import torch
import torch.nn as nn
import os 

from config import Config
from student import StudentEncoder
from teacher import TeacherNetwork
#  
from aggregator_crossview import DepressionAggregator

### 消融
# from aggregator_crossview import DepressionAggregator_no_mu as DepressionAggregator
# from aggregator_crossview import DepressionAggregator_no_crossproj as DepressionAggregator
# from aggregator_crossview import DepressionAggregator_no_stimgate as DepressionAggregator


class ECAN(nn.Module):
    """
    End-to-End 模型封装。
    
    """

    def __init__(self, pretrained_student_path=None, use_teacher=True, freeze_student=True, noise_z=False):
        super(ECAN, self).__init__()
        
        self.use_teacher = use_teacher
        self.freeze_student = freeze_student
        self.noise_z = noise_z  # 是否将 Z_student 替换为 N(0,1) 噪声（消融：验证情绪表征有效性）
        
        # 实例化子模块
        self.student = StudentEncoder()
        self.aggregator = DepressionAggregator()

        # Teacher 
        if use_teacher:
            self.teacher = TeacherNetwork()
            t_state_dict = torch.load(os.path.join(Config.SAVE_DIR, 'teacher_phase1_best.pth'))
            # for param select
            # t_state_dict = torch.load('/home/zhanghb/MyPaperProgram/MyPaper/EmoGeneralization/code_v14/checkpoints/l_kld_0.025/teacher_phase1_best.pth')
            self.teacher.load_state_dict(t_state_dict, strict=False)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        
        # Phase 2训练好的 Student权重加载
        if pretrained_student_path:
            self.student.load_state_dict(torch.load(pretrained_student_path), strict=True)
            print(f"Loaded pretrained Student from {pretrained_student_path}")
        else:
            print('No Student Model Loaded (random init)!')
        
        # 决定 Student 是否参与训练
        if freeze_student:
            for param in self.student.parameters():
                param.requires_grad = False 
            print('Student frozen.')
        else:
            print('Student trainable (end-to-end).')

    def forward(self, audio_feats, text_feats, stim_embeds, mask):
        """
        End-to-End 前向传播 
        
        Input Shapes (来自 PublicDataset):
            audio_feats: (Batch, N_max, 1, D_aud)
            text_feats:  (Batch, N_max, 1, D_txt)
            stim_embeds: (Batch, N_max, 768)
            mask:        (Batch, N_max) 
        """
        batch_size, n_max = stim_embeds.size(0), stim_embeds.size(1)
        # ==========================================
        # Step 1: Flattening (展平) -> Micro Level
        # ==========================================
        flat_audio = audio_feats.view(batch_size * n_max, 1, -1)
        flat_text = text_feats.view(batch_size * n_max, 1, -1)
        flat_stim = stim_embeds.view(batch_size * n_max, -1)
        flat_mask = mask.view(batch_size * n_max, -1)
        
        # ==========================================
        # Step 2: Student Processing
        # ==========================================
        # freeze_student=True: 关闭梯度，eval 模式
        # freeze_student=False: 端到端训练，Student 参与反向传播
        if self.freeze_student:
            with torch.no_grad():
                self.student.eval() 
                s_z, s_mu, s_logvar, student_cls_out = self.student(flat_audio, flat_text, flat_stim, flat_mask) # mu shape: (B*N, Z_Dim)
        else:
            s_z, s_mu, s_logvar, student_cls_out = self.student(flat_audio, flat_text, flat_stim, flat_mask)
        
        # Teacher.decode 生成 recon_emos
        if self.use_teacher:
            with torch.no_grad():
                recon_emos = self.teacher.decode(z=s_z, stim_embeds=flat_stim, emo_seqs=None, teacher_forcing_ratio=0.0)   # shape:(B*N, emo_len, 7)
            recon_emos = recon_emos.view(batch_size, n_max, -1, 7)  # shape:(B, N, emo_len, 7)
        else:
            recon_emos = None  
        
        # 对student_cls_out进行处理
        student_cls_out = student_cls_out.view(batch_size, n_max, -1)  # shape: [B,N,768]
        # student_cls_out = torch.mean(student_cls_out, dim=1)
        
        # ==========================================
        # Step 3: Reshaping (重组) -> Macro Level
        # ==========================================
        if self.noise_z:
            # 消融：将情绪表征 Z_student 替换为标准正态 N(0,1) 噪声
            # 验证模型是否确实在使用情绪表征（而非把 Z 当作空操作）
            z_seqs = torch.randn_like(s_mu).view(batch_size, n_max, -1)  # (B, N, Z)
        else:
            z_seqs = s_mu.view(batch_size, n_max, -1) # (B, N, Z)
        # print(' Student Mu: ', z_seqs)
        
        # ==========================================
        # Step 4: Aggregation
        # ==========================================
        logits = self.aggregator(z_seqs, recon_emos, audio_feats, text_feats, stim_embeds, mask)
        
        return logits
