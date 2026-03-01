
class configs:
    # ------------------
    # 路径配置 (Paths)
    # ------------------
    # 默认数据路径 (如果没有通过命令行参数指定)
    data_file_path = "100.csv" 
    
    # ------------------
    # 环境与图相关 (Environment & Graph)
    # ------------------
    n_j = 715                      # 任务(工序)数量估计 (Graph Nodes)
    n_m = 5                        # 站位数量 (Stations)
    n_w = 30                        # 工人数量 (Workers)
                                    # 注意：实际任务数由 data_loader 动态加载，此处仅作参考或 Embedding 初始化上界
    
    # ------------------
    # 模型超参数 (Model Hyperparameters)
    # ------------------
    hidden_dim = 128                # 隐藏层维度 (Embedding Size)
    num_gat_layers = 3              # GAT 层数 (Message Passing Depth)
    num_heads = 4                   # 多头注意力头数 (Attention Heads)
    dropout = 0.1                  # Dropout 比率 (防止过拟合)
    
    # 输入特征维度 (根据 environment.py 中的 _get_observation 定义)
    task_feat_dim = 17              # Task Node Input Features [Duration, Status(4), Type(10), Wait(1), Demand(1)]
    worker_feat_dim = 12            # Worker Node Input Features [Efficiency(1), Skills(10), IsFree(1)]
    station_feat_dim = 8            # Station Node Input Features [Load(1), NumTasks(1), Padding(6)]
    
    # ------------------
    # 泛化性与域随机化 (Domain Randomization)
    # ------------------
    randomize_durations = True      # 是否在训练期间开启工时随机扰动
    dur_random_range = 0.2          # 工时扰动幅度 (0.2 表示基础工时的 ±20% 波动)
    
    # ------------------
    # PPO 训练超参数 (PPO Training)
    # ------------------
    lr = 1e-4                       # 初始学习率 (Learning Rate)
    gamma = 0.99                    # 折扣因子 (Discount Factor)
    k_epochs = 4                    # 每次更新循环次数 (PPO Update Epochs)
    eps_clip = 0.2                  # PPO Clip阈值 (e.g. 0.1 ~ 0.2)
    batch_size = 2                 # PPO update batch size (小批量更新)
    
    # [Loss Balancing & Critic Isolation 2026-02-22]
    c_policy = 1.0                  # Policy Loss 权重
    # c_value = 0.5                   # Value Loss 权重 (降低预估误差的主宰)
    c_value = 0.05                  # [缓解高误差] 听从用户建议：大幅降低 Value权重，削弱剧烈震荡对整体优化的干扰
    # [2026-02-27] Reduce Entropy to force network out of the random uniform policy (blindness)
    # c_entropy = 0.02                # Entropy Loss 权重 (提高探索度)
    c_entropy = 0.005                 # 降低 Entropy 强迫网络尽快形成固定且高效的分配路线
    # [Advanced Training Features 2026-02-20]
    accumulation_steps = 2          # 梯度累积步长 (在 batch_size=16 时，实际更新等效为 64)
    gae_lambda = 0.95               # GAE 优势函数的衰减因子 (0~1 之间，越小方差越小、偏差越大)
    use_muon = True                 # 是否使用 Muon 优化器进行 2D 张量的更新
    # [SGDR Learning Rate Schedule]
    sgdr_t0 = 40                    # Cosine Annealing 热重启周期 (建议在多节点大图中延长此至 100+)
    
    # [Training Control Parameters 2026-02-12]
    max_episodes = 200             # 最大训练 Episode 数
    update_every_episodes = 2       # 多少个 Episode 收集一次数据进行 PPO 更新
    eval_freq = 2                  # 多少个 Episode 进行一次评估
    eval_temperature = 0.0         # 评估/推理时的采样温度 (0.0 表示确定的 Argmax 贪婪策略)
    
    # [Learning Rate Schedule]
    lr_warmup_steps = 3           # 学习率预热步数 (Linear Warmup)
    min_lr = 1e-5                   # 最小学习率 (Cosine Annealing 下界)

    # ------------------
    # 日志与监控 (Logging)
    # ------------------
    log_dir = "tf-logs"                # TensorBoard 日志目录
