import matplotlib.pyplot as plt
import numpy as np
import os

class AggregateVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # 定义要关注的情绪维度 (根据你的 config 调整)
        # 假设 3: Happy, 5: Sad
        self.target_emotions = {
            3: "Happy", 
            5: "Sad"
        }
    
    def plot_group_dynamics(self, all_real, all_recon, all_labels, all_stim_ids, epoch):
        """
        绘制群体级动力学对比图 (Mean + Std)
        """
        # 1. 选取几个典型的 Stimulus ID 进行展示 (比如样本量最多的前3个)
        unique_stims, counts = np.unique(all_stim_ids, return_counts=True)
        top_stims = unique_stims[np.argsort(-counts)][:3] # 选前3个最常见的刺激
        
        # 创建画布: 行=Stimulus, 列=Emotion
        rows = len(top_stims)
        cols = len(self.target_emotions)
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), sharey=True)
        
        if rows == 1: axes = axes[np.newaxis, :]
        if cols == 1: axes = axes[:, np.newaxis]

        for i, stim_id in enumerate(top_stims):
            # 筛选当前 Stimulus 的数据
            stim_mask = (all_stim_ids == stim_id)
            
            for j, (emo_idx, emo_name) in enumerate(self.target_emotions.items()):
                ax = axes[i, j]
                
                # --- 绘制 Healthy (Label=0) ---
                self._plot_band(ax, all_real, all_recon, all_labels, stim_mask, 
                                target_label=0, emo_idx=emo_idx, 
                                color='green', label_name='Healthy')

                # --- 绘制 Depressed (Label=1) ---
                self._plot_band(ax, all_real, all_recon, all_labels, stim_mask, 
                                target_label=1, emo_idx=emo_idx, 
                                color='red', label_name='Depressed')
                
                ax.set_title(f"Stim {stim_id} - {emo_name}")
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize='small')
                if j == 0:
                    ax.set_ylabel("Intensity")
                if i == rows - 1:
                    ax.set_xlabel("Time Steps")

        plt.suptitle(f"Epoch {epoch+1}: Group Dynamics (Solid=Mean, Shade=Std)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"epoch_{epoch+1}_aggregate.png"))
        plt.close()

    def _plot_band(self, ax, real, recon, labels, stim_mask, target_label, emo_idx, color, label_name):
        # 组合 Mask: 特定刺激 + 特定人群
        mask = stim_mask & (labels == target_label)
        
        if np.sum(mask) == 0:
            return # 该类别无数据

        # 提取数据 (N, 50, 7) -> (N, 50)
        group_real = real[mask, :, emo_idx]
        group_recon = recon[mask, :, emo_idx]
        
        # 计算统计量
        real_mean = np.mean(group_real, axis=0)
        # real_std = np.std(group_real, axis=0) # 真实数据的波动通常很大，画出来会很乱，这里仅画重构的波动
        
        recon_mean = np.mean(group_recon, axis=0)
        recon_std = np.std(group_recon, axis=0)
        
        x = np.arange(len(real_mean))
        
        # 1. 画真实均值 (虚线) - 作为 Ground Truth 参考
        ax.plot(x, real_mean, color=color, linestyle='--', alpha=0.5, linewidth=1)
        
        # 2. 画重构均值 (实线)
        ax.plot(x, recon_mean, color=color, label=f'{label_name} (Recon)', linewidth=2)
        
        # 3. 画重构波动 (阴影) - 这能看出模型是否学到了多样性
        ax.fill_between(x, recon_mean - recon_std, recon_mean + recon_std, 
                        color=color, alpha=0.15)