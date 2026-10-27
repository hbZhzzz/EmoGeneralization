import torch
import os

class Config:
    # ================= 路径配置 =================
    # 数据处理为 .pt 文件
    Project = 'EmoGeneralization/code'
    PRIVATE_DATA_PATH_ALL = os.path.join(Project, 'data/private_data.pt')
    PRIVATE_DATA_PATH_TRAIN = os.path.join(Project, 'data/private_data_train.pt')
    PRIVATE_DATA_PATH_TEST = os.path.join(Project, 'data/private_data_test.pt')

    DATASET_NMAE = 'midd' 
    PUBLIC_DATA_PATH = os.path.join(Project, f'data/public_data_{DATASET_NMAE}.pt')
    
    # SAVE_DIR = './checkpoints/'
    SAVE_DIR = os.path.join(Project, 'checkpoints/')
    CROSSSTUMU_SAVE_DIR = os.path.join(Project, 'run_crossstimu/checkpoints/')
    CROSSDATA_SAVE_DIR = os.path.join(Project, 'run_crossdataset/checkpoints/')
    # 消融实验保存路径
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
    MAX_EMO_LEN = 100     # 教师网络：情绪序列最大帧数
    
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

    # midd:18, modma:9  (跨刺激：midd:3 modma:6) (跨数据集：cmdc=modma=midd=9)
    CurrentStimus = None  
    
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
    DROPOUT = 0.5
    N_HEAD = 1
    TSF_LAYERS = 1 # transformer 组件的层数
    
    # ================= 损失函数权重 =================
    ## teachaer loss 
    SCALE_MSE_TEACHER = 100.0 
    SCALE_MSE_STUDENT = 100.0 

 
    W_CONTRA = 5.0 #default= 5.0
    KLD_WEIGHT = 0.05  
    KLD_START_EPOCH = 5 
    KLD_ANNEAL_EPOCHS = 20 
    FREE_BITS = 0.1


    #phase3 
    COEFFI_ALIGN = 0.01 # 默认=0.01

    # ================= 训练超参数 =================
    BATCH_SIZE = 32     
    LR_STUDENT = 1e-4    
    LR_TEACHER = 5e-4     
    LR_AGGREGATOR = {
        'modma': 3e-3,  
        'midd': 2e-4,  
        'daic': 8e-4,
        'cmdc': 6e-4 
    }

    EPOCHS_WARMUP = 50
    EPOCHS_STUDENT = 100
    EPOCHS_JOINT = 100
    Max_patience = 100 
    
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


