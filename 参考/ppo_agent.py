import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import math

class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体。
    负责与 Environment 交互，收集轨迹，并更新 Strategy Network。
    """
    def __init__(self, model, lr, gamma, k_epochs, eps_clip, device, batch_size=4, 
                 lr_warmup_steps=0, min_lr=0, total_timesteps=0):
        self.policy = model.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.lr = lr
        self.gamma = gamma          # 折扣因子
        self.k_epochs = k_epochs    # 每次 Update 的迭代轮数
        self.eps_clip = eps_clip    # PPO Clip参数 (e.g., 0.2)
        self.device = device
        self.batch_size = batch_size
        
        self.MseLoss = nn.MSELoss()
        
        # [LR Scheduler Setup]
        # Linear Warmup + Cosine Annealing
        self.lr_warmup_steps = lr_warmup_steps
        self.min_lr = min_lr
        self.total_timesteps = max(1, total_timesteps) # 防止除零
        self.current_step = 0
        
        # 定义 LambdaLR
        def lr_lambda(current_step):
            # 1. Warmup Phase
            if current_step < self.lr_warmup_steps:
                return float(current_step) / float(max(1, self.lr_warmup_steps))
            
            # 2. Cosine Decay Phase
            progress = float(current_step - self.lr_warmup_steps) / float(max(1, self.total_timesteps - self.lr_warmup_steps))
            progress = min(1.0, max(0.0, progress)) # Clamp [0, 1]
            
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Scaling: range [min_lr/lr, 1.0]
            min_ratio = self.min_lr / self.lr
            return min_ratio + (1.0 - min_ratio) * cosine_decay
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def select_action(self, obs, mask_task=None, mask_station_matrix=None, mask_worker=None, deterministic=False):
        """
        选择动作 (Select Action)。
        
        Args:
            obs: 异构图观测数据 (HeteroData)
            mask_task: [N] Bool Tensor, True=Invalid
            mask_station_matrix: [N, S] Bool Tensor, True=Invalid
            mask_worker: [W] Bool Tensor, True=Invalid (Global)
            deterministic: 是否确定性选择 (ArgMax vs Sampling)
            
        Returns:
            action_tuple: (task_id, station_id, team_indices_list)
            action_logprob: float
            state_value: float
            specific_station_mask: 用于 Memory 记录
        """
        with torch.no_grad():
            x_dict, global_context = self.policy(obs)
            
            # ------------------
            # 1. 选择工序 (Select Task)
            # ------------------
            task_logits = self.policy.task_head(x_dict['task'], global_context, mask=mask_task)
            
            # [Robustness] 检查并处理 NaN
            if torch.isnan(task_logits).any():
                task_logits = torch.nan_to_num(task_logits, nan=-1e9)
            
            if deterministic:
                if mask_task is not None:
                    task_logits = task_logits.masked_fill(mask_task, -1e9)
                task_action = torch.argmax(task_logits)
                task_logprob = torch.tensor(0.0).to(self.device)
            else:
                if mask_task is not None:
                     task_logits = task_logits.masked_fill(mask_task, -1e9)
                
                # Check for all -inf
                if (task_logits <= -1e8).all():
                     print("WARNING: All Task Logits -inf in select_action. Force picking 0.")
                     task_action = torch.tensor(0).to(self.device)
                     task_logprob = torch.tensor(0.0).to(self.device)
                else:
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
            station_logits = self.policy.station_head(selected_task_emb, station_embs, mask=specific_station_mask)
            
            if torch.isnan(station_logits).any():
                station_logits = torch.nan_to_num(station_logits, nan=-1e9)
            
            if deterministic:
                if specific_station_mask is not None:
                     station_logits = station_logits.masked_fill(specific_station_mask, -1e9)
                station_action = torch.argmax(station_logits)
                station_logprob = torch.tensor(0.0).to(self.device)
            else:
                if specific_station_mask is not None:
                     station_logits = station_logits.masked_fill(specific_station_mask, -1e9)
                
                if (station_logits <= -1e8).all():
                     print("WARNING: All Station Logits -inf. Force picking 0.")
                     station_action = torch.tensor(0).to(self.device)
                     station_logprob = torch.tensor(0.0).to(self.device)
                else:
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
            
            current_worker_mask = current_worker_mask | skill_mask.to(self.device)
            
            worker_embs = x_dict['worker'].unsqueeze(0)
            
            for _ in range(demand):
                # 还有可选工人吗?
                if current_worker_mask.all():
                    break
                
                worker_logits = self.policy.worker_head.forward_choice(selected_task_emb, worker_embs, mask=current_worker_mask)
                
                if torch.isnan(worker_logits).any():
                    worker_logits = torch.nan_to_num(worker_logits, nan=-1e9)
                
                if deterministic:
                     worker_logits = worker_logits.masked_fill(current_worker_mask, -1e9)
                     if (worker_logits <= -1e8).all(): break
                     
                     w_action = torch.argmax(worker_logits)
                     w_lp = torch.tensor(0.0).to(self.device)
                else:
                     worker_logits = worker_logits.masked_fill(current_worker_mask, -1e9)
                     
                     if (worker_logits <= -1e8).all():
                         break # 无法继续选人
                         
                     w_dist = Categorical(logits=worker_logits)
                     w_action = w_dist.sample()
                     w_lp = w_dist.log_prob(w_action)
                
                w_idx = w_action.item()
                team_indices.append(w_idx)
                worker_logprobs.append(w_lp)
                
                # 更新 Mask (选过的人不能再选)
                current_worker_mask = current_worker_mask.clone() # 确保不 原地修改 影响下一轮
                current_worker_mask[w_idx] = True
            
            total_worker_logprob = sum(worker_logprobs) if worker_logprobs else torch.tensor(0.0).to(self.device)
            
            action_logprob = task_logprob + station_logprob + total_worker_logprob
            state_value = self.policy.get_value(global_context)
            
            action_tuple = (t_idx, station_action.item(), team_indices)
            
        return action_tuple, action_logprob.item(), state_value.item(), specific_station_mask

    def update(self, memory):
        """
        PPO 更新逻辑。
        
        Args:
            memory: 存储轨迹的 Buffer
            
        Returns:
            metrics: dict, 用于 TensorBoard 记录
        """
        # 1. 计算 Monte Carlo 折扣回报 (Rewards-to-Go)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).detach()
        # 奖励归一化
        if rewards.std() > 1e-7:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            rewards = rewards - rewards.mean()
        
        # 2. 准备 Batch 数据
        old_actions = memory.actions 
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(self.device)
        
        # Pad Team List (变长 -> 定长 Tensor)
        max_team_size = max(len(a[2]) for a in old_actions) if old_actions else 1
        
        b_task = torch.tensor([a[0] for a in old_actions], dtype=torch.long).to(self.device)
        b_station = torch.tensor([a[1] for a in old_actions], dtype=torch.long).to(self.device)
        
        team_list = []
        for a in old_actions:
            t = a[2]
            pad = [-1] * (max_team_size - len(t))
            team_list.append(t + pad)
        b_team = torch.tensor(team_list, dtype=torch.long).to(self.device)
        
        # Attach targets to Data objects for Batching
        for i, state in enumerate(memory.states):
            state.y_task = b_task[i].unsqueeze(0)
            state.y_station = b_station[i].unsqueeze(0)
            state.y_team = b_team[i].unsqueeze(0) 
            state.y_logprob = old_logprobs[i].unsqueeze(0)
            state.y_reward = rewards[i].unsqueeze(0)
            
            if i < len(memory.masks):
                t_mask, s_mask, w_mask = memory.masks[i]
                state.y_task_mask = t_mask
                state.y_station_mask = s_mask
                state.y_worker_mask = w_mask
        
        loader = DataLoader(memory.states, batch_size=self.batch_size, shuffle=True)
        
        # 3. PPO Optimization Loop
        print(f"PPO Update: BatchSize={self.batch_size}, Total Batches={len(loader)}")
        
        avg_loss = 0
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0
        update_counts = 0
        
        for i_epoch in range(self.k_epochs):
            for batch in loader:
                batch = batch.to(self.device)
                
                # 当前策略的前向传播
                x_dict, global_context = self.policy(batch)
                state_values = self.policy.get_value(global_context).view(-1)
                
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
                
                station_logits = self.policy.station_head(sel_task_emb, station_x, mask=None)
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
                
                for k in range(batch.y_team.size(1)):
                    target = batch.y_team[:, k] 
                    valid_step = (target != -1)
                    if not valid_step.any(): continue
                    
                    logits = self.policy.worker_head.forward_choice(sel_task_emb, worker_x, mask=curr_mask)
                    if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-1e9)
                    
                    dist = Categorical(logits=logits)
                    step_lp = dist.log_prob(torch.clamp(target, min=0)) 
                    team_lp[valid_step] += step_lp[valid_step]
                    entropy[valid_step] += dist.entropy()[valid_step]
                    
                    # Update mask for next worker in team
                    for b in range(target.size(0)):
                        if valid_step[b]:
                            curr_mask = curr_mask.clone()
                            curr_mask[b, target[b]] = True
                            
                total_lp = task_lp + station_lp + team_lp
                
                # --- PPO Loss Calculation ---
                ratios = torch.exp(total_lp - batch.y_logprob)
                adv = batch.y_reward - state_values.detach()
                
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * self.MseLoss(state_values, batch.y_reward)
                entropy_loss = -0.01 * entropy.mean()
                
                loss = policy_loss + value_loss + entropy_loss
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                
                # [Gradient Clipping]
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                # Log Stats
                avg_loss += loss.item()
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy_loss += entropy_loss.item()
                update_counts += 1
        
        # [Step Scheduler]
        self.scheduler.step()
        self.current_step += 1
                
        metrics = {
            'Loss/Total': avg_loss / update_counts if update_counts > 0 else 0,
            'Loss/Policy': avg_policy_loss / update_counts if update_counts > 0 else 0,
            'Loss/Value': avg_value_loss / update_counts if update_counts > 0 else 0,
            'Loss/Entropy': avg_entropy_loss / update_counts if update_counts > 0 else 0,
            'Train/LearningRate': self.scheduler.get_last_lr()[0] # Log Current LR
        }
        return metrics
