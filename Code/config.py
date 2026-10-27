import torch
import os

class Config:
    # ================= 路径配置 =================
    # 假设你的数据都处理成了 .pt 文件
    Project = '/home/zhanghb/MyPaperProgram/MyPaper/EmoGeneralization/code_v14'
    PRIVATE_DATA_PATH_ALL = os.path.join(Project, 'data/private_data_yjq.pt')
    PRIVATE_DATA_PATH_TRAIN = os.path.join(Project, 'data/private_data_yjq_train.pt')
    PRIVATE_DATA_PATH_TEST = os.path.join(Project, 'data/private_data_yjq_test.pt')

    DATASET_NMAE = 'midd'  # 仅在同域测试实验内有效，验证时也要修改
    PUBLIC_DATA_PATH = os.path.join(Project, f'data/public_data_{DATASET_NMAE}.pt')
    
    # SAVE_DIR = './checkpoints/'
    SAVE_DIR = os.path.join(Project, 'checkpoints/')
    CROSSSTUMU_SAVE_DIR = os.path.join(Project, 'run_crossstimu/checkpoints/')
    CROSSDATA_SAVE_DIR = os.path.join(Project, 'run_crossdataset/checkpoints/')
    # 消融实验（无私有数据预训练）专用保存路径，避免覆盖原模型
    ABLATION_SAVE_DIR = os.path.join(Project, 'run_ablation/checkpoints/')
    
    # 输出信息控制 
    AGG_LAYER_PRINT= True
    DETAILA_SAVEA_NAME = True




    # ================= 模型超参数 =================
    SEED = 2026
    n_splits = 5 # k fold
    # 基础维度
    D_MODEL = 768        # BERT/ 的输出维度
    Z_DIM = 64           # 潜变量维度
    EMO_DIM = 7          # 情绪维度 (7类基本情绪)
    AGG_DIM = 128        # 128 (聚合层的隐藏维度，通常比 Z 大一点)
    # 序列相关
    MAX_EMO_LEN = 100     # 教师网络：情绪序列最大帧数 (超过截断，不足补0) 100, 50, 150， 200
    # 注意：音频/文本 T=1，不需要设定序列长度
    MAX_STIMULI = {
        'modma': 18,
        'midd':9,
        'daic':14,
        'cmdc':12
    }
    STIM_DIM = 768       # Stim特征维度
    AUDIO_DIM = 88 
    TEXT_DIM = 768
    NUM_CLASSES = 2

    # =================================================
    # 注意，不同的任务（综合性能测试，跨数据集，跨刺激需要分别设置）
    # midd:18, modma:9  (跨刺激：midd:3 modma:6) (跨数据集：cmdc=modma=midd=9)
    # 适配aggregator_crossview中的a,b分类器，需要手动设置。
    CurrentStimus = None  
    # =================================================

    # 学习率优化
    Start_f = 0.1
    End_f = 1.0
    Total_iters=10
    T_0 = 20
    T_mult = 1 # default = 2
    Milestones = [10]


    # 架构细节
    # 分类网络参数
    AGG_FEED_FORWARD = 256  # 1024 
    AGG_HIDDIM = 256 # 256
    AGG_N_HEAD = 1
    AGG_TSF_LAYERS = 1
    AGG_DROPOUT = 0.5 # 0.5s

    # Teacher-Student网络参数
    TEACHER_LAYERS = 2
    STUDENT_LAYERS = 4
    FEED_FORWARD = 256
    
    # AGG_LAYERS = 1
    DROPOUT = 0.5  # 对小样本数据集，该值可以适当调大（0.3-0.5）
    N_HEAD = 1
    TSF_LAYERS = 1 # transformer 组件的层数
    
    # ================= 损失函数权重 =================
    ## teachaer loss 
    # 100 MSE衡量不同情绪的概率是否相似，但大部分情绪概率的范围波动很小，例如，预测 0.10 和预测 0.15 的 MSE 差距是 0.05^2 = 0.0025
    # 这个梯度太小了，网络根本“感觉”不到误差，反向传播时梯度在几层之后就消失了。
    # 因此，我们为MSE添加一个数值尺度放大系数 SCALE_MSE， 但在显示loss时，我们不显示这个SCALE_MSE，否则loss会看起来很大
    SCALE_MSE_TEACHER = 100.0 
    SCALE_MSE_STUDENT = 100.0 # 100，modma-acc=0.98， f1=0.98, 10 modma-acc=0.96

 
    W_CONTRA = 5.0 #default= 5.0
    KLD_WEIGHT = 0.05  #  phase1阶段，default= 0.05
    KLD_START_EPOCH = 5 # phase1中，计算KL损失的退火epoch数, 在该epoch前，kl权重为0.0。
    KLD_ANNEAL_EPOCHS = 20 
    FREE_BITS = 0.1


    #phase3 
    COEFFI_ALIGN = 0.01 # 默认=0.01

    # ================= 训练超参数 =================
    BATCH_SIZE = 32      # 样本级 Batch Size
    LR_STUDENT = 1e-4   # 
    LR_TEACHER = 5e-4     # 
    LR_AGGREGATOR = {
        'modma': 3e-3,  # 
        'midd': 2e-4,  # 6e-4， 5e-4， 1e-4（0.84，0.86）3e-4(0.83,0.86) 9e-5(0.84,0.86)
        'daic': 8e-4,
        'cmdc': 6e-4 # 3e-4 8e-4(0.9091)# 4e-3(0.88) 6e-4(0.9216)
    }

    EPOCHS_WARMUP = 50
    EPOCHS_STUDENT = 100
    EPOCHS_JOINT = 100
    Max_patience = 100 # 早停耐心值
    
    Lr_Scheduler = False 

    # ================= 设备 =================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    EMO_CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    EMO_CLASSES_NUM = len(EMO_CLASS_NAMES)


    @staticmethod
    def print_config():
        print(f"Device: {Config.DEVICE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Model Dim: {Config.D_MODEL}, Z Dim: {Config.Z_DIM}")

    @classmethod
    def get_datapath(cls, dataset_name, CrossStimu=False, CrossData=False):

        if CrossStimu:
            return os.path.join(cls.Project, f'data/cross_stimu/public_data_{dataset_name}.pt')
        elif CrossData:
            return os.path.join(cls.Project, f'data/cross_dataset/public_data_{dataset_name}_cross_dataset.pt')
        else:
            return os.path.join(cls.Project, f'data/public_data_{dataset_name}.pt')


