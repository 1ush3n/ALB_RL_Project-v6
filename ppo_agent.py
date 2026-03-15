import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import math
from configs import configs
from utils.muon import Muon
class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体。
    负责与 Environment 交互，收集轨迹，并更新 Strategy Network。
    """
    def __init__(self, model, lr, gamma, k_epochs, eps_clip, device, batch_size=4, 
                 lr_warmup_steps=0, min_lr=0, total_timesteps=0):
        self.policy = model.to(device)
        
        # [Optimizer Setup: Muon + AdamW]
        if hasattr(configs, 'use_muon') and configs.use_muon:
            muon_params = []
            adam_params = []
            for name, param in self.policy.named_parameters():
                if param.ndim >= 2:
                    muon_params.append(param)
                else:
                    adam_params.append(param)
                    
            self.optimizer = Muon(muon_params, lr=lr * 0.02, momentum=0.95)
            self.optimizer_adam = torch.optim.AdamW(adam_params, lr=lr)
            self.using_muon = True
        else:
            self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4) # Modified to AdamW
            self.using_muon = False
            
        self.lr = lr
        self.gamma = gamma          # 折扣因子
        self.k_epochs = k_epochs    # 每次 Update 的迭代轮数
        self.eps_clip = eps_clip    # PPO Clip参数 (e.g., 0.2)
        self.device = device
        self.batch_size = batch_size
        self.accumulation_steps = getattr(configs, 'accumulation_steps', 1)
        self.gae_lambda = getattr(configs, 'gae_lambda', 0.95)
        
        self.MseLoss = nn.MSELoss() # 回归标准 MSE，强制 Critic 网络具有针对大数值误差的抛物线追赶能力
        
        self.target_kl = getattr(configs, 'target_kl', 0.015)
        self.min_lr = getattr(configs, 'min_lr', 1e-6)
        self.lr_max = getattr(configs, 'lr_max', 5e-4)
        self.lr_warmup_steps = lr_warmup_steps
        self.initial_lr = lr
        
        self.total_timesteps = max(1, total_timesteps)
        self.current_step = 0
        
        # [CRITICAL FIX 2026-03-13] 我们彻底废除了预设规律的衰减引擎 (如 SGDR )。
        # SGDR (Cosine Annealing) 会在周期到达时对学习率进行跃迁重启，这极大破坏了 PPO 的近端策略约束 (Trust Region)。
        # 接下来我们将依照用户建议，使用动态评估前后新旧策略差距 (KL散度) 的自适应方法直接在 update 尾部变动 LR。
    def select_action(self, obs, mask_task=None, mask_station_matrix=None, mask_worker=None, deterministic=False, temperature=1.0, is_eval=False):
        """
        选择动作 (Select Action)。
        
        Args:
            obs: 异构图观测数据 (HeteroData)
            mask_task: [N] Bool Tensor, True=Invalid
            mask_station_matrix: [N, S] Bool Tensor, True=Invalid
            mask_worker: [W] Bool Tensor, True=Invalid (Global)
            deterministic: 是否确定性选择 (ArgMax vs Sampling)
            temperature: 采样温度，T越小越贪婪，T越大越随机，忽略当 deterministic=True 时
            
        Returns:
            action_tuple: (task_id, station_id, team_indices_list)
            action_logprob: float
            state_value: float
            specific_station_mask: 用于 Memory 记录
        """
        from configs import configs
        no_mask = getattr(configs, 'ablation_no_mask', False)
        
        with torch.no_grad():
            x_dict, global_context = self.policy(obs)
            
            # [Phase 1: Robustness] 获取动态适配的 dtype 极小值（防止 FP16 下 -1e9 溢出）
            mask_value = torch.finfo(x_dict['task'].dtype).min / 2.0
            
            # ------------------
            # 1. 选择工序 (Select Task)
            # ------------------
            task_logits = self.policy.task_head(x_dict['task'], global_context, mask=mask_task if not no_mask else None)
            
            # [Robustness] 检查并处理 NaN
            if torch.isnan(task_logits).any():
                task_logits = torch.nan_to_num(task_logits, nan=mask_value)
            
            if deterministic:
                if mask_task is not None and not no_mask:
                    task_logits = task_logits.masked_fill(mask_task, mask_value)
                task_action = torch.argmax(task_logits)
                task_logprob = torch.tensor(0.0).to(self.device)
            else:
                if mask_task is not None and not no_mask:
                     task_logits = task_logits.masked_fill(mask_task, mask_value)
                
                # Check for all -inf
                if (task_logits <= mask_value * 0.99).all():
                     print("WARNING: All Task Logits -inf in select_action. Force picking 0.")
                     task_action = torch.tensor(0).to(self.device)
                     task_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        task_logits = task_logits / max(temperature, 1e-5)
                    task_dist = Categorical(logits=task_logits)
                    task_action = task_dist.sample()
                    task_logprob = task_dist.log_prob(task_action)
            
            t_idx = task_action.item()
            selected_task_emb = x_dict['task'][t_idx].unsqueeze(0) # [1, H]
            
            # 获取任务的人数需求
            raw_demand = obs['task'].x[t_idx, -1].item()
            demand = int(raw_demand)
            if demand < 1: demand = 1 # Safety clamp
            
            # ------------------
            # 2. 选择站位 (Select Station)
            # ------------------
            specific_station_mask = None
            if mask_station_matrix is not None:
                # [N, S] -> [1, S]
                specific_station_mask = mask_station_matrix[t_idx].unsqueeze(0)
            
            station_embs = x_dict['station'].unsqueeze(0)
            station_logits = self.policy.station_head(selected_task_emb, station_embs, mask=specific_station_mask if not no_mask else None)
            
            if torch.isnan(station_logits).any():
                station_logits = torch.nan_to_num(station_logits, nan=mask_value)
            
            if deterministic:
                if specific_station_mask is not None and not no_mask:
                     station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                station_action = torch.argmax(station_logits)
                station_logprob = torch.tensor(0.0).to(self.device)
            else:
                if specific_station_mask is not None and not no_mask:
                     station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                
                if (station_logits <= mask_value * 0.99).all():
                     print("WARNING: All Station Logits -inf. Force picking 0.")
                     station_action = torch.tensor(0).to(self.device)
                     station_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        station_logits = station_logits / max(temperature, 1e-5)
                    station_dist = Categorical(logits=station_logits)
                    station_action = station_dist.sample()
                    station_logprob = station_dist.log_prob(station_action)
                
            # ------------------
            # 3. 选择工人 (Select Workers) - 自回归
            # ------------------
            team_indices = []
            worker_logprobs = []
            
            # 动态 Mask: 初始 Mask + 技能 Mask
            current_worker_mask = mask_worker.clone() if mask_worker is not None else torch.zeros(obs['worker'].num_nodes, dtype=torch.bool).to(self.device)
            
            worker_feats = obs['worker'].x
            worker_skills = worker_feats[:, 1:11] # 10 dim
            
            task_type_idx = torch.argmax(obs['task'].x[t_idx, 5:15]).item() 
            
            has_skill = worker_skills[:, task_type_idx] > 0.5
            skill_mask = ~has_skill 
            
            s_act = station_action.item() + 1
            worker_locks = torch.argmax(worker_feats[:, 13:21], dim=1)
            lock_mask = (worker_locks != 0) & (worker_locks != s_act)
            
            # [Hybrid Masking] 2. 稀缺技能保护掩码 (Skill Preservation Mask)
            preservation_mask = torch.zeros(obs['worker'].num_nodes, dtype=torch.bool).to(self.device)
            unlocked_mask = (worker_locks == 0)
            
            for k in range(10):
                if k == task_type_idx: continue
                # 散人中懂得技能 k 的总人数
                workers_with_k = worker_skills[:, k] > 0.5
                free_experts_k = (workers_with_k & unlocked_mask).sum().item()
                # 如果散人库里的 k 专家 <= 2 人 (即独苗濒危)
                if free_experts_k <= 2:
                    # 强行保护：如果你懂这门频危手艺但当前任务却不需要这门手艺，那你不能参加！
                    is_valuable_expert = workers_with_k & unlocked_mask 
                    preservation_mask = preservation_mask | is_valuable_expert.to(self.device)
            
            if no_mask:
                current_worker_mask = skill_mask.to(self.device)
            else:
                # 试算：应用保护掩码后是否还有足够人手满足 demand
                trial_mask = current_worker_mask | skill_mask.to(self.device) | lock_mask.to(self.device) | preservation_mask
                if (~trial_mask).sum().item() >= demand:
                    current_worker_mask = trial_mask # 启用独苗保护
                else:
                    # 活儿干不完了，动用战略储备
                    current_worker_mask = current_worker_mask | skill_mask.to(self.device) | lock_mask.to(self.device)
            
            worker_embs = x_dict['worker'].unsqueeze(0)
            
            # [Phase 1: Robustness] 加入迭代阈值和 Fallback 防止因掩码过度重叠发生死循环
            max_iter = demand * 2
            iter_cnt = 0
            
            # [Phase 5: Autoregressive Optimization A] 初始化团队记忆
            current_team_emb = None 
            
            while len(team_indices) < demand and iter_cnt < max_iter:
                iter_cnt += 1
                
                # 还有可选工人吗?
                if current_worker_mask.all():
                    break
                
                worker_logits = self.policy.worker_head.forward_choice(selected_task_emb, worker_embs, mask=current_worker_mask, current_team_emb=current_team_emb)
                
                if torch.isnan(worker_logits).any():
                    worker_logits = torch.nan_to_num(worker_logits, nan=mask_value)
                
                if deterministic:
                     if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                     if (worker_logits <= mask_value * 0.99).all(): break
                     
                     w_action = torch.argmax(worker_logits)
                     w_lp = torch.tensor(0.0).to(self.device)
                else:
                     if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                     
                     if (worker_logits <= mask_value * 0.99).all():
                         break # 无法继续选人
                     
                     if temperature != 1.0:
                         worker_logits = worker_logits / max(temperature, 1e-5)
                         
                     w_dist = Categorical(logits=worker_logits)
                     w_action = w_dist.sample()
                     w_lp = w_dist.log_prob(w_action)
                
                w_idx = w_action.item()
                team_indices.append(w_idx)
                worker_logprobs.append(w_lp)
                
                # [Phase 5: Autoregressive Optimization A] 刷新已选团队表征记忆
                selected_worker_feats = worker_embs[0, team_indices, :]
                current_team_emb = selected_worker_feats.mean(dim=0, keepdim=True) # [1, H]
                
                # 更新 Mask (选过的人不能再选)
                current_worker_mask = current_worker_mask.clone() # 确保不 原地修改 影响下一轮
                current_worker_mask[w_idx] = True
            
            # [兜底逻辑] 若因过度竞争或死锁选不够人选
            if len(team_indices) < demand:
                if is_eval:
                    # [Evaluation Strict Mode] 验证期间绝对不允许兜底作弊！
                    # 如果选不够人，说明策略出现断层死锁，直接将失败上传以施加真实的验证集惩罚。
                    return None, 0.0, 0.0, None, True
                    
                # [Zero-Fallback Enforcement] 原有的兜底机制已被彻底移除。
                # 由于环境的 get_masks() 已经在物理和拓扑层面上保证了只有当满足 demand 人数（且技能、工位锁定状态都符合要求）时，
                # 站位和任务才是合法的。如果在这里选不出足够的人，说明前置掩码与内层选人掩码存在逻辑脱节，或出现了未知的计算漏洞。
                # 此时绝不可再凑数塞入假人或存入假概率，这会导致后期 update 产生爆炸的虚假 KL 并诱发一连串的崩溃！
                raise RuntimeError(
                    f"FATAL DEADLOCK: Failed to select enough valid workers (needed {demand}, got {len(team_indices)}).\n"
                    f"The masking logic in environment get_masks() strictly guarantees worker sufficiency.\n"
                    f"No manual fallback is ever allowed to preserve the KL purity. Please inspect the mask consistency!"
                )
            
            
            total_worker_logprob = sum(worker_logprobs) if worker_logprobs else torch.tensor(0.0).to(self.device)
            
            action_logprob = task_logprob + station_logprob + total_worker_logprob
            # [CRITICAL FIX] 物理隔离隔离 Critic 防止其巨大的 Value Error 梯度捣毁底层共享 GAT 拓扑特征 (灾难性干扰致盲)
            # [Phase 6: Dual-Stream Critic Evaluation]
            # 传入完整的 state (batch_data)，由于处于 with torch.no_grad() 下，此处无需 detach，直接前向提取价值。
            state_value = self.policy.get_value(obs)
            
            action_tuple = (t_idx, station_action.item(), team_indices)
            
            # [Phase 5] Check validity of action for soft penalty
            is_invalid_action = False
            if mask_task is not None and mask_task[t_idx].item():
                is_invalid_action = True
            if specific_station_mask is not None and specific_station_mask[0, station_action.item()].item():
                is_invalid_action = True
            if mask_worker is not None:
                for w_idx in team_indices:
                    if mask_worker[w_idx].item():
                        is_invalid_action = True
            
        return action_tuple, action_logprob.item(), state_value.item(), specific_station_mask, is_invalid_action

    def update(self, memory, env=None):
        """
        PPO 更新逻辑。
        
        Args:
            memory: 存储轨迹的 Buffer
            
        Returns:
            metrics: dict, 用于 TensorBoard 记录
        """
        # 1. 计算广义优势估计 (GAE - Generalized Advantage Estimation)
        rewards = []
        advantages = []
        gae = 0
        
        # 将 rewards 与 values 张量化以进行 GAE 计算
        mem_rewards = memory.rewards
        mem_is_terminals = memory.is_terminals
        
        # 提取存储在 states 中的 state_values
        # (这需要在 select_action 之后被记录下来，如果没有记录，回退为普通的 MC 回报加基线)
        if hasattr(memory, 'values') and len(memory.values) == len(mem_rewards):
            mem_values = memory.values
            next_value = 0 # 终止状态后的 value 为 0
            
            for step in reversed(range(len(mem_rewards))):
                if mem_is_terminals[step]:
                    next_value = 0
                    gae = 0
                
                delta = mem_rewards[step] + self.gamma * next_value - mem_values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages.insert(0, gae)
                next_value = mem_values[step]
                
            advantages = torch.tensor(advantages, dtype=torch.float32)
            rewards = advantages + torch.tensor(mem_values, dtype=torch.float32)
        else:
            # Fallback 到 Monte-Carlo + Advantage (如果缺少 Value 记录)
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(mem_rewards), reversed(mem_is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            rewards = torch.tensor(rewards, dtype=torch.float32)
            # 兼容处理
            advantages = rewards.clone()
            
        # 归一化 Advantages 与 Returns (有助于长期负反馈环境的训练稳定性)
        if advantages.std() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        else:
            advantages = advantages - advantages.mean()
            
        # [CRITICAL FIX: Removed Return Normalization]
        # 绝不应对 Critic 的 Target Returns 进行动态批次标准化，
        # 否则每一轮 Update 的均值和方差都在变（移动靶），导致 Critic 永远无法收敛，产生巨大的梯度震荡。
        # 我们改用配置中的静态系数缩小全局 reward。
        
        # 2. 准备 Batch 数据
        old_actions = memory.actions 
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        
        # Pad Team List (变长 -> 定长 Tensor)
        max_team_size = max(len(a[2]) for a in old_actions) if old_actions else 1
        
        b_task = torch.tensor([a[0] for a in old_actions], dtype=torch.long)
        b_station = torch.tensor([a[1] for a in old_actions], dtype=torch.long)
        
        team_list = []
        for a in old_actions:
            t = a[2]
            pad = [-1] * (max_team_size - len(t))
            team_list.append(t + pad)
        b_team = torch.tensor(team_list, dtype=torch.long)
        
        # Attach targets to Data objects for Batching
        rebuilt_states = []
        if env is not None:
             for snap in memory.states:
                 rebuilt_states.append(env.rebuild_state_from_snapshot(snap))
        else:
             rebuilt_states = memory.states
             
        for i, state in enumerate(rebuilt_states):
            state.y_task = b_task[i].unsqueeze(0)
            state.y_station = b_station[i].unsqueeze(0)
            state.y_team = b_team[i].unsqueeze(0) 
            state.y_logprob = old_logprobs[i].unsqueeze(0)
            state.y_reward = rewards[i].unsqueeze(0)
            state.y_advantage = advantages[i].unsqueeze(0)
            
            # [Added] Load original state values for PPO Value Clipping
            if len(memory.values) > i:
                 state.y_value = torch.tensor([memory.values[i]], dtype=torch.float32)
            
            if i < len(memory.masks):
                t_mask, s_mask, w_mask = memory.masks[i]
                state.y_task_mask = t_mask
                state.y_station_mask = s_mask
                state.y_worker_mask = w_mask
        
        loader = DataLoader(rebuilt_states, batch_size=self.batch_size, shuffle=True)
        
        # 3. PPO Optimization Loop
        print(f"PPO Update: BatchSize={self.batch_size}, Total Batches={len(loader)}")
        
        avg_loss = 0
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0
        update_counts = 0
        approx_kls = []
        
        self.optimizer.zero_grad()
        if self.using_muon:
            self.optimizer_adam.zero_grad()
            
        final_epoch = self.k_epochs - 1
        kl_meltdown = False # [Phase 8: KL Armor]
        
        for i_epoch in range(self.k_epochs):
            if kl_meltdown: break
            
            epoch_kls = []
            for step_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                
                # 当前策略的前向传播
                x_dict, global_context = self.policy(batch)
                
                # [CRITICAL Phase 6: Dual-Stream Critic Evaluation]
                # 传入完整的 batch，由 Critic 独立的 Embedder 和 Encoder 提取特征，不与 Actor 共享骨干！
                # 并且去除了 .detach()，让梯度真正在 Critic 内部乃至 Attention Pooling 中流动！
                state_values = self.policy.get_value(batch).view(-1)
                
                # --- Re-evaluate LogProbs ---
                # A. Task LogProb
                from torch_geometric.utils import to_dense_batch
                task_x, p_mask = to_dense_batch(x_dict['task'], batch['task'].batch)
                
                # 恢复 Mask
                if hasattr(batch, 'y_task_mask'):
                    logical_task_mask, _ = to_dense_batch(batch.y_task_mask, batch['task'].batch)
                    combined_task_mask = logical_task_mask | (~p_mask)
                else:
                    combined_task_mask = ~p_mask
                    
                task_logits = self.policy.task_head(task_x, global_context, mask=combined_task_mask)
                if torch.isnan(task_logits).any(): task_logits = torch.nan_to_num(task_logits, nan=-1e9)
                
                task_dist = Categorical(logits=task_logits)
                task_lp = task_dist.log_prob(batch.y_task)
                entropy = task_dist.entropy()
                
                # B. Station LogProb
                batch_indices = torch.arange(batch.y_task.size(0)).to(self.device)
                sel_task_emb = task_x[batch_indices, batch.y_task] 
                
                station_x, s_p_mask = to_dense_batch(x_dict['station'], batch['station'].batch)
                
                if hasattr(batch, 'y_station_mask'):
                    dense_s_mask, _ = to_dense_batch(batch.y_station_mask, batch['task'].batch)
                    specific_station_mask = dense_s_mask[batch_indices, batch.y_task]
                    curr_s_mask = specific_station_mask | (~s_p_mask)
                else:
                    curr_s_mask = ~s_p_mask
                
                station_logits = self.policy.station_head(sel_task_emb, station_x, mask=curr_s_mask)
                if torch.isnan(station_logits).any(): station_logits = torch.nan_to_num(station_logits, nan=-1e9)
                
                station_dist = Categorical(logits=station_logits)
                station_lp = station_dist.log_prob(batch.y_station)
                entropy += station_dist.entropy()
                
                # C. Worker Team LogProb
                worker_x, w_p_mask = to_dense_batch(x_dict['worker'], batch['worker'].batch)
                team_lp = torch.zeros_like(task_lp)
                
                if hasattr(batch, 'y_worker_mask'):
                     d_w_mask, _ = to_dense_batch(batch.y_worker_mask.float(), batch['worker'].batch)
                     curr_mask = (d_w_mask > 0.5) | (~w_p_mask)
                else:
                     curr_mask = (~w_p_mask)
                
                # Add Skill Mask based on the selected task
                task_raw, _ = to_dense_batch(batch['task'].x, batch['task'].batch)
                sel_task_raw = task_raw[batch_indices, batch.y_task]
                task_type_idx = torch.argmax(sel_task_raw[:, 5:15], dim=1) # [B]
                
                worker_raw, _ = to_dense_batch(batch['worker'].x, batch['worker'].batch)
                worker_skills = worker_raw[:, :, 1:11] # [B, Max_W, 10]
                
                B_size, Max_W_size = worker_skills.shape[0], worker_skills.shape[1]
                b_indices_expanded = torch.arange(B_size).view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                w_indices_expanded = torch.arange(Max_W_size).view(1, -1).expand(B_size, -1).reshape(-1)
                t_indices_expanded = task_type_idx.view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                
                has_skill_flat = worker_skills[b_indices_expanded, w_indices_expanded, t_indices_expanded] > 0.5
                skill_mask = (~has_skill_flat).view(B_size, Max_W_size).to(self.device)
                
                s_act = batch.y_station + 1 # [B]
                worker_locks = torch.argmax(worker_raw[:, :, 13:21], dim=2) # [B, Max_W]
                s_act_expanded = s_act.view(B_size, 1).expand(B_size, Max_W_size).to(self.device)
                lock_mask = (worker_locks != 0) & (worker_locks != s_act_expanded)
                
                curr_mask = curr_mask | skill_mask | lock_mask.to(self.device)
                
                # [Phase 5: Autoregressive Optimization A] 
                current_team_emb = None # [B, H]
                team_emb_sum = torch.zeros(B_size, worker_x.size(-1)).to(self.device)
                team_cnt = torch.zeros(B_size, 1).to(self.device)
                
                for k in range(batch.y_team.size(1)):
                    target = batch.y_team[:, k] 
                    valid_step = (target != -1)
                    if not valid_step.any(): continue
                    
                    logits = self.policy.worker_head.forward_choice(sel_task_emb, worker_x, mask=curr_mask, current_team_emb=current_team_emb)
                    if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-1e9)
                    
                    dist = Categorical(logits=logits)
                    step_lp = dist.log_prob(torch.clamp(target, min=0)) 
                    team_lp[valid_step] += step_lp[valid_step]
                    entropy[valid_step] += dist.entropy()[valid_step]
                    
                    # [Phase 5: Update current_team_emb for next iteration inside graph execution mode]
                    valid_b_indices = torch.nonzero(valid_step).squeeze(-1)
                    valid_targets = target[valid_step]
                    
                    selected_feats = worker_x[valid_b_indices, valid_targets]
                    
                    # 使用 clone() 保障 PyTorch 自动求导机制的连续性 (Gradient Preservation)
                    next_team_emb_sum = team_emb_sum.clone()
                    next_team_cnt = team_cnt.clone()
                    
                    next_team_emb_sum[valid_b_indices] += selected_feats
                    next_team_cnt[valid_b_indices] += 1
                    
                    team_emb_sum = next_team_emb_sum
                    team_cnt = next_team_cnt
                    
                    current_team_emb = team_emb_sum / torch.clamp(team_cnt, min=1.0)
                    
                    # Update mask for next worker in team
                    curr_mask = curr_mask.clone()
                    curr_mask[valid_b_indices, target[valid_step]] = True
                            
                total_lp = task_lp + station_lp + team_lp
                
                # [Phase 2: LogProb Clipping] 防止后续的 torch.exp 发生指数散度爆炸
                total_lp = torch.clamp(total_lp, min=-20.0, max=2.0)
                
                # --- PPO Loss Calculation ---
                with torch.no_grad():
                    log_ratio = total_lp - batch.y_logprob.view(-1)
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    epoch_kls.append(approx_kl.item())
                # [Phase 8: Inner-Batch KL Meltdown Protection]
                if approx_kl.item() > getattr(configs, 'kl_early_stop', 0.03):
                    print(f"      [EMERGENCY STOP] Batch {step_idx}: KL={approx_kl.item():.4f} > Break Threshold. Aborting PPO update to protect Policy.")
                    kl_meltdown = True
                    break
                ratios = torch.exp(total_lp - batch.y_logprob.view(-1))
                
                # Use GAE advantages if available, else batch.y_reward - state_values (MC fallback)
                b_adv = batch.y_advantage.view(-1) if hasattr(batch, 'y_advantage') else (batch.y_reward.view(-1) - state_values.detach())
                
                # [Phase 2: Dynamic EPS Clip & Entropy Annealing] 动态衰减探索上限
                progress = min(1.0, self.current_step / max(1, self.total_timesteps))
                curr_eps_clip = self.eps_clip - progress * (self.eps_clip - 0.05)
                
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1-curr_eps_clip, 1+curr_eps_clip) * b_adv
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value and Entropy Loss scaled by configs
                c_val = getattr(configs, 'c_value', 0.5)
                c_ent_base = getattr(configs, 'c_entropy', 0.01)
                # [Phase 2: Entropy Decay 修复] 受控的熵退火，随进程向贪婪坍缩。
                # 依据 configs.py 中指定的 entropy_decay_episodes 来精确计算退火进度
                decay_eps = getattr(configs, 'entropy_decay_episodes', 200)
                update_freq = getattr(configs, 'update_every_episodes', 2)
                decay_updates = max(1, decay_eps // update_freq)  # 将 Episode 跨度转换为 Update 次数跨度
                
                ent_progress = min(1.0, self.current_step / decay_updates)
                
                c_ent_base = getattr(configs, 'c_entropy', 0.05)
                c_ent_end = getattr(configs, 'c_entropy_end', 0.01)
                # [Phase 8: Exponential Entropy Decay]
                # 指数衰减 (比线性更平滑，在初期下降快，后期保留长久的微弱探索尾巴)
                import math
                c_ent = c_ent_end + (c_ent_base - c_ent_end) * math.exp(-3.0 * ent_progress)
                
                c_pol = getattr(configs, 'c_policy', 1.0)
                
                # [Phase 8: Critic Armor (Huber Loss / SmoothL1 Loss)]
                b_reward = batch.y_reward.view(-1)
                
                # 为了防止死锁的 -50 满额惩罚在单纯的 MSE 中引发上千的核爆均方梯度的击穿，
                # 改用 SmoothL1Loss，对 < beta 的温和误差使用平方，对 > beta 的极端误差使用绝对值削平。
                value_loss = c_val * F.smooth_l1_loss(state_values, b_reward, beta=5.0)
                     
                entropy_loss = -c_ent * entropy.mean()
                
                loss = c_pol * policy_loss + value_loss + entropy_loss
                
                # Backprop
                loss = loss / self.accumulation_steps # 归一化 Gradient
                loss.backward()
                
                # [Gradient Accumulation]
                if ((step_idx + 1) % self.accumulation_steps == 0) or (step_idx + 1 == len(loader)):
                    # [Phase 8: Independent Gradient Clipping]
                    # 分别搜集 Actor 和 Critic 的参数，由于双流网络已解绑，Critic 对大梯度的耐受力远弱于 Actor。
                    actor_params = [p for n, p in self.policy.named_parameters() if 'critic' not in n and 'attn' not in n]
                    critic_params = [p for n, p in self.policy.named_parameters() if 'critic' in n or 'attn' in n]
                    
                    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
                    # 给 Critic 挂装远比 Actor 更薄弱的装甲 (0.1)，防止局部脉冲带崩全盘
                    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=getattr(configs, 'clip_v_grad_norm', 0.1))
                    
                    self.optimizer.step()
                    if self.using_muon:
                        self.optimizer_adam.step()
                        self.optimizer_adam.zero_grad()
                    self.optimizer.zero_grad()
                    
                    update_counts += 1
                
                # Log Stats (取消除以 accumulation_steps 来显示真实 loss 幅度)
                avg_loss += loss.item() * self.accumulation_steps
                # [Phase 6: Logging Raw, Unscaled Losses (Loss Transparency)]
                avg_policy_loss += policy_loss.item()
                avg_value_loss += F.smooth_l1_loss(state_values, b_reward, beta=5.0).item() # 记录受护甲保护后的真实误差
                avg_entropy_loss += entropy.mean().item() # 记录最本源的策略熵 (不含截断)
            
            # [CRITICAL FIX: Epoch Early Stopping for Trust Region Protection]
            # 计算当前 epoch 的平均 KL
            curr_epoch_kl = sum(epoch_kls) / len(epoch_kls) if epoch_kls else 0.0
            
            # 我们始终记录最后一轮未掐断的 KL 作为自适应引擎的参考
            approx_kls = epoch_kls
            
            # 如果偏离已经大于设定的容忍度，立刻停止剩下的 Epochs！这极大地缩短了耗时，直接修好了假死！
            if curr_epoch_kl > 1.5 * self.target_kl:
                print(f"      -> Early stopping at epoch {i_epoch+1} due to reaching max KL: {curr_epoch_kl:.4f}")
                break
        
        # [Adaptive KL LR Update 2026-03-13] 取代机械定式重启的 SGDR
        mean_kl = sum(approx_kls) / len(approx_kls) if approx_kls else 0.0
        
        if self.current_step < self.lr_warmup_steps:
             # [Phase 2: Linear Warmup 2026-03-13] 预热期内保护性放大 LR
             warmup_factor = float(self.current_step + 1) / float(max(1, self.lr_warmup_steps))
             target_lr = self.initial_lr * warmup_factor
             for param_group in (self.optimizer_adam.param_groups if self.using_muon else self.optimizer.param_groups):
                 param_group['lr'] = target_lr
             if self.using_muon:
                 for param_group in self.optimizer.param_groups:
                     param_group['lr'] = target_lr * 0.02
        else:
            if mean_kl > self.target_kl * 1.5:
                 # 近端策略变化太过激进：紧急收紧下调学习率
                 for param_group in (self.optimizer_adam.param_groups if self.using_muon else self.optimizer.param_groups):
                     param_group['lr'] = max(param_group['lr'] / 1.05, self.min_lr)
                 if self.using_muon:
                     for param_group in self.optimizer.param_groups:
                         param_group['lr'] = max(param_group['lr'] / 1.05, self.min_lr * 0.02)
            elif mean_kl < self.target_kl / 1.5:
                 # 近端策略迟滞不前：提速放宽学习率
                 for param_group in (self.optimizer_adam.param_groups if self.using_muon else self.optimizer.param_groups):
                     param_group['lr'] = min(param_group['lr'] * 1.05, self.lr_max)
                 if self.using_muon:
                     for param_group in self.optimizer.param_groups:
                         param_group['lr'] = min(param_group['lr'] * 1.05, self.lr_max * 0.02)
                     
        self.current_step += 1
                
        metrics = {
            'Loss/Total': avg_loss / update_counts if update_counts > 0 else 0,
            'Loss/Policy': avg_policy_loss / update_counts if update_counts > 0 else 0,
            'Loss/Value': avg_value_loss / update_counts if update_counts > 0 else 0,
            'Loss/Entropy': avg_entropy_loss / update_counts if update_counts > 0 else 0,
            'Loss/ApproxKL': mean_kl,
            'Train/LearningRate': self.optimizer_adam.param_groups[0]['lr'] if self.using_muon else self.optimizer.param_groups[0]['lr']
        }
        return metrics
