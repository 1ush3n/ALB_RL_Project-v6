
class configs:
    # ------------------
    # 路径配置 (Paths)
    # ------------------
    # 默认数据路径 (如果没有通过命令行参数指定)
    data_dir = "data"
    data_file_path = "data/100.csv" 
    worker_pool_path = "data/worker_pool_fixed.csv"
    
    # ------------------
    # 环境与图相关 (Environment & Graph)
    # ------------------
    n_j = 3300                       # 任务(工序)数量估计 (Graph Nodes)
    n_m = 5                         # 站位数量 (Stations)
    n_w_max = 120                   # 工人池总上限 (最大可配置的工人数量，固定池容量)
    n_w_min = 50                    # 每回合训练随机抽取的最小工人数 (Domain Randomization)
    n_w = 100                        # 每回合训练抽取的最大工人数，及验证(Eval)阶段固定的工人数
                                    # 注意：实际任务数由 data_loader 动态加载，此处仅作参考或 Embedding 初始化上界
    max_station_capacity_ratio = 0.4  # [Hybrid Masking] 单个站位最大容许绑定全厂总人数的比例，超过此值则站位被强制 Mask 屏蔽
    max_slots_per_station = 3        # [Slot Model] 每站位同时执行的最大工序数（物理工位槽），满槽后新工序需等待
    
    # ------------------
    # 模型超参数 (Model Hyperparameters)
    # ------------------
    hidden_dim = 128                # 隐藏层维度 (Embedding Size)
    num_gat_layers = 5              # GAT 层数 (Message Passing Depth)
    num_heads = 4                   # 多头注意力头数 (Attention Heads)
    dropout = 0.1                  # Dropout 比率 (防止过拟合)
    
    # 输入特征维度 (根据 environment.py 中的 _get_observation 定义)
    task_feat_dim = 17              # Task Node Input Features [Duration, Status(4), Type(10), Wait(1), Demand(1)]
    worker_feat_dim = 21            # Worker Node Input Features [Efficiency(1), Skills(10), is_free(1), ProjectedWait(1), Lock(8)]
    station_feat_dim = 13           # Station Node Input Features [Load(1), BoundRatio(1), MobileRatio(1), FreeBoundRatio(1), SlotWait(1)]
    
    # ------------------
    # 泛化性与域随机化 (Domain Randomization)
    # ------------------
    randomize_durations = True      # 是否在训练期间开启工时随机扰动
    dur_random_range = 0.2          # 工时扰动幅度 (0.2 表示基础工时的 ±20% 波动)
    curriculum_episodes = 500       # [课程式学习] 训练前 N 轮强制关闭所有随机因子，稳定 Critic 拟合
    
    # ------------------
    # PPO 训练超参数 (PPO Training)
    # ------------------
    # [Temperature Annealing] 行动温度退火机制
    temp_start = 2.0                # 初期高温度：鼓励广泛探索各站位和离谱策略组合
    temp_end = 0.1                  # 后期低温度：收拢为贪心，精练出局部最佳解
    temp_decay_episodes = 2000      # 经过多少 Episode 退火降至 temp_end
    
    lr = 1e-4                       # 初始学习率 (3000节点序列极长，不可轻易放大以免陷入局部最优死坑)
    gamma = 0.9995                  # [治病良方：时间折现因子] 3000步超级长线，必须将远视能力拉满！(1 / (1-0.9995) = 2000步视野)
    k_epochs = 4                    # 每次更新循环次数 (从 2 上调至 4，由于当前 KL 极小说明每次榨取的数据不足，应该对同批数据多更新几次来逼近目标 KL)
    eps_clip = 0.2                  # PPO Clip阈值 (e.g. 0.1 ~ 0.2)
    batch_size = 2                 # [防 OOM + 性能提升] 适当放宽至 8 兼顾 8G 显存稳定性
    
    # [Loss Balancing & Critic Isolation 2026-02-22]
    c_policy = 1.0                  # Policy Loss 权重
    c_value = 0.05                   # [已通过 Huber Loss 防爆] 安全调回标准的 0.5，Critic 不会再破坏全局梯度
    
    # [Reward Coefficients 2026-03-12]
    r_coef_makespan = 1.0           # 宏观目标：Makespan 下班时间推移惩罚 (极其容易稀疏，因为只看瓶颈)
    # [Hotfix 2026-03-13] 彻底抹除导致贪婪短视的全局稠密化误导惩罚，严格遵从学术界 GAE 全局溯回理论
    r_coef_wait     = 0.0           # (已废除) 微操目标：工序受到的绝对排队折磨时长 
    # [Target Returns Scaler 2026-03-13]
    reward_scale    = 0.005         # 全局奖励缩放乘数：在 environment 层面将上几千分差的值域缩放到 [-5, 5] 内，极大地稳定 Critic 的方差
    
    # [针对 3000 单的长序列防死锁补丁] 面对巨量状态，初期随机性非常关键。不可过低。
    c_entropy = 0.05                
    # [Entropy Annealing 2026-03-11] 强制智能体在后期收紧探索，不要摆烂
    c_entropy_end = 0.01            # 熵衰减的终点 (上调以免重蹈后期 Policy 极端自信后崩溃引发 Loss 的覆辙)
    entropy_decay_episodes = 200    # 用多少代将 c_entropy 从初始值平滑降至 c_entropy_end
    accumulation_steps = 128       # [防过拟合核心机制] 在内存中聚集高达 16*128=2048 步全局经验后才做 1 次 PPO Update！严防过快更新导致跌入“死磕前几个节点”的局部最优！
    gae_lambda = 0.98               # GAE 优势函数的衰减因子 (适配 3000 极长序列，将长期优势传导给前置任务)
    use_muon = True                 # 是否使用 Muon 优化器进行 2D 张量的更新
    # [SGDR Learning Rate Schedule]
    sgdr_t0 = 150                   # 针对多节点大图大幅延长重启周期 (150 ep 一个深空潜航)
    
    # [Training Control Parameters 2026-02-12]
    max_episodes = 3000             # 探索万亿级组合的三千大劫
    update_every_episodes = 2       # 多少个 Episode 收集一次数据进行 PPO 更新
    eval_freq = 2                  # 多少个 Episode 进行一次评估
    eval_temperature = 0.0         # [Hotfix 2026-03-13] 必须为 0.0！评估/推理时的采样温度 (0.0 表示确定的 Argmax 贪婪策略，1.0 意味着纯随机掷骰子)
    
    # [Adaptive KL Learning Rate Schedule 2026-03-13]
    target_kl = 0.015               # 策略目标 KL 散度（衡量新旧策略差异的距离指标），用于自适应学习率调整。
    lr_warmup_steps = 3             # 学习率预热步数 (Linear Warmup)
    min_lr = 1e-6                   # 最小学习率保护下界
    lr_max = 1e-3                   # 最大学习率保护上界 (放宽至 1e-3 以应对 KL 长时间极度低迷的情况)
    
    # ------------------
    # 日志与监控 (Logging)
    # ------------------
    log_dir = "tf-logs"                # TensorBoard 日志目录
