
class configs:
    # ------------------
    # 路径配置 (Paths)
    # ------------------
    # 默认数据路径 (如果没有通过命令行参数指定)
    data_file_path = "3000.csv" 
    
    # ------------------
    # 环境与图相关 (Environment & Graph)
    # ------------------
    n_j = 3300                      # 任务(工序)数量估计 (Graph Nodes)
    n_m = 20                        # 站位数量 (Stations)
    n_w = 60                        # 工人数量 (Workers)
                                    # 注意：实际任务数由 data_loader 动态加载，此处仅作参考或 Embedding 初始化上界
    
    # ------------------
    # 模型超参数 (Model Hyperparameters)
    # ------------------
    hidden_dim = 128                # 隐藏层维度 (Embedding Size)
    num_gat_layers = 5              # GAT 层数 (Message Passing Depth)
    num_heads = 4                   # 多头注意力头数 (Attention Heads)
    dropout = 0.12                  # Dropout 比率 (防止过拟合)
    
    # 输入特征维度 (根据 environment.py 中的 _get_observation 定义)
    task_feat_dim = 17              # Task Node Input Features [Duration, Status(4), Type(10), Wait(1), Demand(1)]
    worker_feat_dim = 12            # Worker Node Input Features [Efficiency(1), Skills(10), IsFree(1)]
    station_feat_dim = 8            # Station Node Input Features [Load(1), NumTasks(1), Padding(6)]
    
    # ------------------
    # PPO 训练超参数 (PPO Training)
    # ------------------
    lr = 3e-5                       # 初始学习率 (Learning Rate)
    gamma = 0.99                    # 折扣因子 (Discount Factor)
    k_epochs = 1                    # 每次更新循环次数 (PPO Update Epochs)
    eps_clip = 0.2                  # PPO Clip阈值 (e.g. 0.1 ~ 0.2)
    batch_size = 2                  # PPO update batch size (小批量更新)
    
    # [Training Control Parameters 2026-02-12]
    max_episodes = 2000             # 最大训练 Episode 数
    update_every_episodes = 2       # 多少个 Episode 收集一次数据进行 PPO 更新
    eval_freq = 10                  # 多少个 Episode 进行一次评估
    
    # [Learning Rate Schedule]
    lr_warmup_steps = 300           # 学习率预热步数 (Linear Warmup)
    min_lr = 1e-6                   # 最小学习率 (Cosine Annealing 下界)

    # ------------------
    # 日志与监控 (Logging)
    # ------------------
    log_dir = "tf-logs"                # TensorBoard 日志目录
