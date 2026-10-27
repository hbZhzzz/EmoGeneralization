import torch

def get_group_statistics(student_model, dataloader, device):
    """
    计算 '抑郁组' 和 '健康组' 在 Latent Space 的统计特征 (原型)。
    """
    student_model.eval()
    
    # 存储所有的 mu
    depressed_mus = []
    healthy_mus = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取 Student 预测的分布中心 (mu)
            # 注意：这里我们不需要 logvar，因为我们要统计的是"不同人之间的差异"，
            # 而不是"单个人预测的不确定性"。
            _, mu, _ = student_model(
                batch['audio'].to(device), 
                batch['text'].to(device), 
                batch['stim'].to(device)
            )
            
            labels = batch['labels'].to(device)
            
            # 根据标签分组
            for i in range(len(labels)):
                if labels[i] == 1: # 抑郁
                    depressed_mus.append(mu[i])
                else: # 健康
                    healthy_mus.append(mu[i])
    
    # 转换为 Tensor 并计算均值和标准差
    # 结果 shape: (64,)
    stats = {}
    if depressed_mus:
        d_stack = torch.stack(depressed_mus)
        stats['dep'] = {
            'mu': torch.mean(d_stack, dim=0),
            'std': torch.std(d_stack, dim=0) # 这里捕获了群体的多样性
        }
    
    if healthy_mus:
        h_stack = torch.stack(healthy_mus)
        stats['healthy'] = {
            'mu': torch.mean(h_stack, dim=0),
            'std': torch.std(h_stack, dim=0)
        }
        
    return stats

def generate_group_samples(teacher_model, stim_embeds, group_stats, num_samples=50):
    """
    生成特定群体的样本。
    参数:
        group_stats: 上面函数算出来的 stats['dep'] 或 stats['healthy']
    """
    # 1. 扩充 Stimulus: (1, 768) -> (50, 768)
    # 假设 stim_embeds 是单个刺激向量
    batch_stim = stim_embeds.unsqueeze(0).expand(num_samples, -1)
    
    # 2. 准备分布参数: (64,) -> (50, 64)
    mu_group = group_stats['mu'].unsqueeze(0).expand(num_samples, -1).to(stim_embeds.device)
    # logvar = 2 * log(std)
    logvar_group = 2 * torch.log(group_stats['std']).unsqueeze(0).expand(num_samples, -1).to(stim_embeds.device)
    
    # 3. 调用 Mode 3 (基于分布采样)
    # 这样 Teacher 会在这个群体的"中心"附近进行随机采样
    generated_seqs = teacher_model.inference(
        stim_embeds=batch_stim, 
        mu=mu_group, 
        logvar=logvar_group, 
        sample=True # <--- 关键：开启采样以获得多样性
    )
    
    return generated_seqs