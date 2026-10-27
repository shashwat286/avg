import argparse
import copy
import os
import pickle
import time
from datetime import datetime

import cv2
import gymnasium as gym
from gymnasium.wrappers import ClipAction, FlattenObservation, NormalizeObservation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from avg import orthogonal_weight_init, human_format_numbers, set_threads
from incremental_rl.td_error_scaler import TDErrorScaler


def calc_norm(param_list, is_grad=False):
    """ Computes L2 norm of parameters or parameter gradients """
    total = 0.0
    for p in param_list:
        if is_grad and p.grad is None:
            continue  # e.g. frozen core
        val = p.grad.data if is_grad else p.data
        total += val.norm(2).item() ** 2
    return total ** 0.5


def build_core(n_hid, n_core_layers):
    """ Shared core: n_core_layers x [Linear(n_hid, n_hid), LeakyReLU]. n_core_layers=1 matches the original layout. """
    layers = []
    for _ in range(n_core_layers):
        layers += [nn.Linear(n_hid, n_hid), nn.LeakyReLU()]
    return nn.Sequential(*layers)


def core_linear_layers(core):
    """ Shared core Linear layers, ordered lowest (next to input heads) -> highest (next to output heads) """
    return [m for m in core if isinstance(m, nn.Linear)]


def core_param_layer_index(core):
    """ For each tensor in core.parameters(), the index of the core layer it belongs to """
    return [l for l, lin in enumerate(core_linear_layers(core)) for _ in lin.parameters()]


def layer_drift(core, anchor_params):
    """ Per-layer L2 distance ||theta_l - theta_l*|| for each shared core layer (lowest first) """
    layer_of = core_param_layer_index(core)
    sq = [0.0] * (max(layer_of) + 1)
    for l, p, a in zip(layer_of, core.parameters(), anchor_params):
        sq[l] += (p - a).norm(2).item() ** 2
    return [s ** 0.5 for s in sq]


def load_stage1_partial(module, opt, saved_sd, saved_opt_sd, saved_param_names):
    """ Load a Stage-1 checkpoint whose task_specs differ from the current ones (e.g. a Humanoid->Hopper
    checkpoint into a Humanoid->Walker2d agent). Shared core + task-1 heads (and their Adam state) are
    copied by parameter NAME; parameters that don't exist / don't match in shape (new task-2 heads) stay
    freshly initialised with empty Adam state, exactly like an untouched head in the original run. """
    cur = module.state_dict()
    keep = {k: v for k, v in saved_sd.items() if k in cur and cur[k].shape == v.shape}
    module.load_state_dict(keep, strict=False)
    old_idx = {n: i for i, n in enumerate(saved_param_names)}
    new_names = [n for n, _ in module.named_parameters()]
    state = {}
    for j, n in enumerate(new_names):
        i = old_idx.get(n)
        if n in keep and i is not None and i in saved_opt_sd['state']:
            state[j] = saved_opt_sd['state'][i]
    group = dict(saved_opt_sd['param_groups'][0])
    group['params'] = list(range(len(new_names)))
    opt.load_state_dict({'state': state, 'param_groups': [group]})
    return sorted(set(cur) - set(keep))


class ContinualActor(nn.Module):
    def __init__(self, task_specs, device, n_hid=256, n_core_layers=1):
        super(ContinualActor, self).__init__()
        self.device = device
        self.n_core_layers = n_core_layers
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.n_hid = n_hid

        self.input_heads = nn.ModuleDict()
        self.mu_heads = nn.ModuleDict()
        self.log_std_heads = nn.ModuleDict()

        for task, (obs_dim, act_dim) in task_specs.items():
            self.input_heads[task] = nn.Sequential(
                nn.Linear(obs_dim, n_hid),
                nn.LeakyReLU(),
            )
            self.mu_heads[task] = nn.Linear(n_hid, act_dim)
            self.log_std_heads[task] = nn.Linear(n_hid, act_dim)

        # Shared Core Representation Backbone
        self.core = build_core(n_hid, n_core_layers)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, task):
        h = self.input_heads[task](obs.to(self.device))
        phi = self.core(h)
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))  # Penultimate Normalization (pnorm)
        mu = self.mu_heads[task](phi)
        log_std = self.log_std_heads[task](phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))
        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)
        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)

        action = torch.tanh(action_pre)
        action_info = {'mu': mu, 'log_std': log_std, 'dist': dist, 'lprob': lprob, 'action_pre': action_pre}
        return action, action_info


class ContinualQ(nn.Module):
    def __init__(self, task_specs, device, n_hid=256, n_core_layers=1):
        super(ContinualQ, self).__init__()
        self.device = device
        self.n_core_layers = n_core_layers

        self.input_heads = nn.ModuleDict()
        self.q_heads = nn.ModuleDict()

        for task, (obs_dim, act_dim) in task_specs.items():
            self.input_heads[task] = nn.Sequential(
                nn.Linear(obs_dim + act_dim, n_hid),
                nn.LeakyReLU(),
            )
            self.q_heads[task] = nn.Linear(n_hid, 1)

        # Shared Core Representation Backbone
        self.core = build_core(n_hid, n_core_layers)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action, task):
        x = torch.cat((obs, action), -1).to(self.device)
        h = self.input_heads[task](x)
        phi = self.core(h)
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))  # Penultimate Normalization (pnorm)
        return self.q_heads[task](phi).view(-1)


class ContinualAVG:
    def __init__(self, task_specs, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.alpha = cfg.alpha_lr
        self.gamma = cfg.gamma
        self.task_specs = task_specs

        self.n_core_layers = getattr(cfg, 'n_core_layers', 1)
        self.actor = ContinualActor(task_specs, device=cfg.device, n_hid=cfg.nhid_actor, n_core_layers=self.n_core_layers)
        self.Q = ContinualQ(task_specs, device=cfg.device, n_hid=cfg.nhid_critic, n_core_layers=self.n_core_layers)

        # EWC anchor (set via set_ewc) and optional online Fisher accumulator (Option B)
        self.ewc = None
        self.online_fisher = None
        # Running sum of the current task's core gradients between probes (Stage-2 interference analysis)
        self.grad_track = None

        self.popt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=cfg.betas)
        self.qopt = torch.optim.Adam(self.Q.parameters(), lr=cfg.critic_lr, betas=cfg.betas)

        # Task-specific scalers (since target return scales differ across tasks)
        self.td_error_scalers = {task: TDErrorScaler() for task in task_specs}
        self.G = {task: 0 for task in task_specs}
        self.steps = 0

    def compute_action(self, obs, task):
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, action_info = self.actor(obs, task)
        return action, action_info

    def update(self, obs, action, next_obs, reward, done, task, record_diagnostics=False, **kwargs):
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        next_obs = torch.Tensor(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, lprob = action.to(self.device), kwargs['lprob']

        # Task return scaling
        r_ent = reward - self.alpha * lprob.detach().item()
        self.G[task] += r_ent
        if done:
            self.td_error_scalers[task].update(reward=r_ent, gamma=0, G=self.G[task])
            self.G[task] = 0
        else:
            self.td_error_scalers[task].update(reward=r_ent, gamma=self.gamma, G=None)

        # Q loss (Action Value Gradient temporal difference)
        q = self.Q(obs, action.detach(), task)
        with torch.no_grad():
            next_action, action_info = self.actor(next_obs, task)
            next_lprob = action_info['lprob']
            q2 = self.Q(next_obs, next_action, task)
            target_V = q2 - self.alpha * next_lprob

        delta = reward + (1 - done) * self.gamma * target_V - q
        delta /= self.td_error_scalers[task].sigma
        qloss = delta ** 2

        # Action Value Gradient (AVG) policy loss
        curr_action, action_info = self.actor(obs, task)
        curr_lprob = action_info['lprob']
        q_pi = self.Q(obs, curr_action, task)
        ploss = self.alpha * curr_lprob - q_pi

        # EWC: layer-wise quadratic anchor on the shared cores. Uses only stored tensors (theta*, F), no replay.
        ewc_a, ewc_q = 0.0, 0.0
        ploss_total, qloss_total = ploss, qloss
        if self.ewc is not None:
            ewc_a_t = self.ewc_penalty('actor')
            ewc_q_t = self.ewc_penalty('critic')
            ploss_total = ploss + ewc_a_t
            qloss_total = qloss + ewc_q_t
            if record_diagnostics:
                ewc_a, ewc_q = ewc_a_t.item(), ewc_q_t.item()

        self.popt.zero_grad()
        ploss_total.backward()
        self._accumulate_online_fisher('actor', self.actor.core, task)
        self._track_grads('actor', self.actor.core, task)

        a_core_gn, a_head_gn = 0.0, 0.0
        if record_diagnostics:
            a_core_gn = calc_norm(self.actor.core.parameters(), is_grad=True)
            a_head_gn = calc_norm([
                *self.actor.input_heads[task].parameters(),
                *self.actor.mu_heads[task].parameters(),
                *self.actor.log_std_heads[task].parameters()
            ], is_grad=True)

        self.popt.step()

        self.qopt.zero_grad()
        qloss_total.backward()
        self._accumulate_online_fisher('critic', self.Q.core, task)
        self._track_grads('critic', self.Q.core, task)

        q_core_gn, q_head_gn = 0.0, 0.0
        if record_diagnostics:
            q_core_gn = calc_norm(self.Q.core.parameters(), is_grad=True)
            q_head_gn = calc_norm([
                *self.Q.input_heads[task].parameters(),
                *self.Q.q_heads[task].parameters()
            ], is_grad=True)

        self.qopt.step()
        self.steps += 1
        if self.online_fisher is not None and task == self.online_fisher['task']:
            self.online_fisher['count'] += 1
        if self.grad_track is not None and task == self.grad_track['task']:
            self.grad_track['count'] += 1

        if record_diagnostics:
            return {
                'a_core_gn': a_core_gn,
                'a_head_gn': a_head_gn,
                'q_core_gn': q_core_gn,
                'q_head_gn': q_head_gn,
                'ploss': ploss.item(),
                'qloss': qloss.item(),
                'sigma': self.td_error_scalers[task].sigma,
                'ewc_a': ewc_a,
                'ewc_q': ewc_q,
            }
        return None

    # ------------------------------------------------------------------
    # EWC utilities
    # ------------------------------------------------------------------
    def _cores(self):
        return {'actor': self.actor.core, 'critic': self.Q.core}

    def set_ewc(self, fisher_actor, fisher_critic, layer_lambdas):
        """ Anchor the shared cores at their current weights with per-layer strengths.
        layer_lambdas[l] is the lambda for core layer l (0 = lowest, next to the input heads). """
        assert len(layer_lambdas) == self.n_core_layers, "need one lambda per core layer"
        fishers = {'actor': fisher_actor, 'critic': fisher_critic}
        self.ewc = {}
        for key, core in self._cores().items():
            layer_of = core_param_layer_index(core)
            self.ewc[key] = {
                'theta_star': [p.detach().clone() for p in core.parameters()],
                'fisher': [f.detach().clone() for f in fishers[key]],
                'lam': [float(layer_lambdas[l]) for l in layer_of],
            }

    def ewc_penalty(self, key):
        """ sum_l (lambda_l / 2) * sum_i F_i (theta_i - theta*_i)^2 over the shared core of `key` """
        e = self.ewc[key]
        total = 0.0
        for p, p_star, f, lam in zip(self._cores()[key].parameters(), e['theta_star'], e['fisher'], e['lam']):
            if lam > 0:
                total = total + 0.5 * lam * (f * (p - p_star) ** 2).sum()
        if not torch.is_tensor(total):
            total = torch.zeros((), device=self.device)
        return total

    def enable_online_fisher(self, task, beta=0.9999):
        """ Option B: EMA of squared core gradients accumulated during training on `task` """
        self.online_fisher = {
            'task': task, 'beta': beta, 'count': 0,
            'actor': [torch.zeros_like(p) for p in self.actor.core.parameters()],
            'critic': [torch.zeros_like(p) for p in self.Q.core.parameters()],
        }

    def _accumulate_online_fisher(self, key, core, task):
        of = self.online_fisher
        if of is None or task != of['task']:
            return
        b = of['beta']
        with torch.no_grad():
            for p, f in zip(core.parameters(), of[key]):
                if p.grad is not None:
                    f.mul_(b).addcmul_(p.grad, p.grad, value=1.0 - b)

    def get_online_fisher(self):
        """ Bias-corrected online Fisher: (actor_list, critic_list) """
        of = self.online_fisher
        if of is None or of['count'] == 0:
            return None
        corr = 1.0 - of['beta'] ** of['count']
        return [f / corr for f in of['actor']], [f / corr for f in of['critic']]

    # ------------------------------------------------------------------
    # Gradient tracking for the Stage-2 interference probe
    # ------------------------------------------------------------------
    def start_grad_track(self, task):
        self.grad_track = {
            'task': task, 'count': 0,
            'actor': [torch.zeros_like(p) for p in self.actor.core.parameters()],
            'critic': [torch.zeros_like(p) for p in self.Q.core.parameters()],
        }

    def _track_grads(self, key, core, task):
        gt = self.grad_track
        if gt is None or task != gt['task']:
            return
        with torch.no_grad():
            for p, s in zip(core.parameters(), gt[key]):
                if p.grad is not None:
                    s.add_(p.grad)

    def pop_grad_track(self):
        """ Mean current-task core gradient since the last call; resets the accumulator """
        gt = self.grad_track
        n = max(gt['count'], 1)
        out = ([s / n for s in gt['actor']], [s / n for s in gt['critic']], gt['count'])
        for s in gt['actor'] + gt['critic']:
            s.zero_()
        gt['count'] = 0
        return out

    def get_parameter_diagnostics(self, task, anchor=None):
        with torch.no_grad():
            a_core_pn = calc_norm(self.actor.core.parameters(), is_grad=False)
            q_core_pn = calc_norm(self.Q.core.parameters(), is_grad=False)
            a_head_pn = calc_norm([
                *self.actor.input_heads[task].parameters(),
                *self.actor.mu_heads[task].parameters(),
                *self.actor.log_std_heads[task].parameters()
            ], is_grad=False)
            q_head_pn = calc_norm([
                *self.Q.input_heads[task].parameters(),
                *self.Q.q_heads[task].parameters()
            ], is_grad=False)

            a_core_drift = 0.0
            q_core_drift = 0.0
            a_layer_drift = [0.0] * self.n_core_layers
            q_layer_drift = [0.0] * self.n_core_layers
            if anchor is not None:
                if 'actor_core' in anchor:
                    a_layer_drift = layer_drift(self.actor.core, anchor['actor_core'])
                    a_core_drift = sum(d ** 2 for d in a_layer_drift) ** 0.5
                if 'critic_core' in anchor:
                    q_layer_drift = layer_drift(self.Q.core, anchor['critic_core'])
                    q_core_drift = sum(d ** 2 for d in q_layer_drift) ** 0.5

        return {
            'a_core_pn': a_core_pn,
            'q_core_pn': q_core_pn,
            'a_head_pn': a_head_pn,
            'q_head_pn': q_head_pn,
            'a_core_drift': a_core_drift,
            'q_core_drift': q_core_drift,
            'a_layer_drift': a_layer_drift,
            'q_layer_drift': q_layer_drift,
        }

    def snapshot_core(self):
        """ Snapshots core weights to compute representation drift in subsequent stages """
        with torch.no_grad():
            return {
                'actor_core': [p.detach().clone() for p in self.actor.core.parameters()],
                'critic_core': [p.detach().clone() for p in self.Q.core.parameters()],
            }

    def save(self, model_path, norm_envs=None, extra=None):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
            "task_specs": self.task_specs,
            "n_core_layers": self.n_core_layers,
            "nhid_actor": self.cfg.nhid_actor,
            "nhid_critic": self.cfg.nhid_critic,
            "norm_stats": {},
        }
        if extra:
            model.update(extra)
        if norm_envs:
            for task, env_wrapper in norm_envs.items():
                if hasattr(env_wrapper, "obs_rms"):
                    model["norm_stats"][task] = {
                        "mean": env_wrapper.obs_rms.mean,
                        "var": env_wrapper.obs_rms.var,
                        "count": env_wrapper.obs_rms.count,
                    }
        torch.save(model, model_path)


# ----------------------------------------------------------------------
# Replay-free Fisher estimation for the shared cores
# ----------------------------------------------------------------------
def fisher_from_rollout(agent, env_name, env, norm_env, n_steps=5000):
    """ Option A: stream n_steps of task transitions (no training, nothing stored).
    Actor  : F = E[(d log pi(a|s) / d theta)^2], a ~ pi   (true policy Fisher)
    Critic : F = E[(d Q(s,a) / d theta)^2]                (Gauss-Newton Fisher for the squared TD loss) """
    saved_rms = copy.deepcopy(norm_env.obs_rms)  # do not let the rollout shift Stage-1 normalizer stats
    norm_env.update_running_mean = False
    F_a = [torch.zeros_like(p) for p in agent.actor.core.parameters()]
    F_q = [torch.zeros_like(p) for p in agent.Q.core.parameters()]
    obs, _ = env.reset()
    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)

        agent.actor.zero_grad()
        action, info = agent.actor(obs_t, env_name)
        info['lprob'].sum().backward()
        for p, f in zip(agent.actor.core.parameters(), F_a):
            f.add_(p.grad.detach() ** 2)

        agent.Q.zero_grad()
        agent.Q(obs_t, action.detach(), env_name).sum().backward()
        for p, f in zip(agent.Q.core.parameters(), F_q):
            f.add_(p.grad.detach() ** 2)

        obs, _, term, trunc, _ = env.step(action.detach().cpu().view(-1).numpy())
        if term or trunc:
            obs, _ = env.reset()
    agent.actor.zero_grad()
    agent.Q.zero_grad()
    norm_env.obs_rms = saved_rms
    norm_env.update_running_mean = True
    return [f / n_steps for f in F_a], [f / n_steps for f in F_q]


def fisher_from_adam(agent):
    """ Option C: Adam's bias-corrected second moment (EMA of squared loss gradients, beta2 = 0.999) """
    out = []
    for opt, core in ((agent.popt, agent.actor.core), (agent.qopt, agent.Q.core)):
        beta2 = opt.param_groups[0]['betas'][1]
        fl = []
        for p in core.parameters():
            st = opt.state[p]
            step = float(st['step'])
            fl.append(st['exp_avg_sq'].detach().clone() / (1.0 - beta2 ** step))
        out.append(fl)
    return out[0], out[1]


def normalize_fisher(fl):
    """ Scale so the mean Fisher value over the whole core is 1 (makes lambda comparable across nets/methods) """
    mean = torch.cat([f.flatten() for f in fl]).mean().clamp_min(1e-12)
    return [f / mean for f in fl]


def fisher_layer_means(core, fl):
    layer_of = core_param_layer_index(core)
    sums, counts = {}, {}
    for l, f in zip(layer_of, fl):
        sums[l] = sums.get(l, 0.0) + f.sum().item()
        counts[l] = counts.get(l, 0) + f.numel()
    return [sums[l] / counts[l] for l in sorted(sums)]


def evaluate_task(agent, env_name, norm_env, episodes=20, device="cpu", action_mode="deterministic"):
    raw_env = gym.make(env_name)
    flat_env = FlattenObservation(raw_env)
    eval_norm_env = NormalizeObservation(flat_env)
    eval_env = ClipAction(eval_norm_env)

    # Copy observation statistics
    if norm_env and hasattr(norm_env, "obs_rms"):
        eval_norm_env.obs_rms.mean = copy.deepcopy(norm_env.obs_rms.mean)
        eval_norm_env.obs_rms.var = copy.deepcopy(norm_env.obs_rms.var)
        eval_norm_env.obs_rms.count = norm_env.obs_rms.count
        eval_norm_env.update_running_mean = False  # gymnasium>=1.0 freeze flag

    rets, steps_list = [], []
    for ep in range(episodes):
        obs, _ = eval_env.reset(seed=ep + 1000)
        ret, step = 0, 0
        term, trunc = False, False
        while not (term or trunc):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action, action_info = agent.actor(obs_t, env_name)
                if action_mode == "deterministic":
                    act = torch.tanh(action_info['mu']).cpu().view(-1).numpy()
                else:
                    act = action.cpu().view(-1).numpy()
            obs, reward, term, trunc, _ = eval_env.step(act)
            ret += reward
            step += 1
        rets.append(ret)
        steps_list.append(step)
    eval_env.close()
    return np.mean(rets), np.std(rets) / np.sqrt(len(rets)), np.mean(steps_list)


# ----------------------------------------------------------------------
# Stage-2 gradient interference probe (replay-free)
# ----------------------------------------------------------------------
def _flat(tl):
    return torch.cat([t.flatten() for t in tl])


def _cos(a, b):
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


class InterferenceProbe:
    """ Every probe_freq Stage-2 steps:
      g_T : mean core gradient of the NEW task's losses over the updates since the last probe (what drove the updates)
      g_H : core gradient of the OLD task's (Humanoid) losses on freshly streamed Humanoid transitions (never applied)
      cos(g_T, g_H)      > 0: descending the new task also descends Humanoid (transfer)
                         < 0: descending the new task climbs Humanoid's loss (interference)
      dL_pred = g_H . dtheta : first-order predicted change in Humanoid loss caused by the actual core updates
                               since the last probe (> 0 = forgetting pressure)
      plus deterministic Humanoid eval return (the forgetting curve) """

    def __init__(self, agent, old_task, norm_env_old, csv_path, probe_steps=300, eval_episodes=3, seed=0):
        self.agent, self.old_task, self.norm_env_old = agent, old_task, norm_env_old
        self.csv_path, self.probe_steps, self.eval_episodes = csv_path, probe_steps, eval_episodes
        self.n_layers = agent.n_core_layers

        env = gym.make(old_task)
        env = FlattenObservation(env)
        self.norm = NormalizeObservation(env)
        self.norm.obs_rms = copy.deepcopy(norm_env_old.obs_rms)
        self.norm.update_running_mean = False
        self.env = ClipAction(self.norm)
        self.env.reset(seed=seed)

        self.prev_a = [p.detach().clone() for p in agent.actor.core.parameters()]
        self.prev_q = [p.detach().clone() for p in agent.Q.core.parameters()]
        self.layer_of = core_param_layer_index(agent.actor.core)

        cols = ["step", "n_updates", "h_eval_ret", "h_eval_len", "h_ploss", "h_qloss"]
        for k in ("act", "crit"):
            cols += [f"{k}_gH_norm", f"{k}_gT_norm", f"{k}_cos", f"{k}_upd_norm", f"{k}_upd_cos_gH", f"{k}_dL_pred"]
        for k in ("act", "crit"):
            cols += [f"{k}_cos_L{l}" for l in range(self.n_layers)]
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")

    def _old_task_grads(self):
        ag, task = self.agent, self.old_task
        core_params = list(ag.actor.core.parameters()) + list(ag.Q.core.parameters())
        was_frozen = [not p.requires_grad for p in core_params]
        for p in core_params:
            p.requires_grad_(True)  # measure Humanoid's core gradient even if the core is frozen
        try:
            return self._old_task_grads_inner()
        finally:
            for p, fr in zip(core_params, was_frozen):
                if fr:
                    p.requires_grad_(False)
                    p.grad = None

    def _old_task_grads_inner(self):
        ag, task = self.agent, self.old_task
        gA = [torch.zeros_like(p) for p in ag.actor.core.parameters()]
        gQ = [torch.zeros_like(p) for p in ag.Q.core.parameters()]
        sigma = ag.td_error_scalers[task].sigma
        pl_sum, ql_sum = 0.0, 0.0
        obs, _ = self.env.reset()
        for _ in range(self.probe_steps):
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(ag.device)
            # Actor: same AVG policy loss as training
            ag.actor.zero_grad(); ag.Q.zero_grad()
            a, info = ag.actor(o, task)
            pl = (ag.alpha * info['lprob'] - ag.Q(o, a, task)).sum()
            pl.backward()
            for p, g in zip(ag.actor.core.parameters(), gA):
                g.add_(p.grad)
            # Critic: same scaled TD loss as training
            next_obs, r, term, trunc, _ = self.env.step(a.detach().cpu().view(-1).numpy())
            ag.Q.zero_grad()
            q = ag.Q(o, a.detach(), task)
            with torch.no_grad():
                no = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(ag.device)
                na, ninfo = ag.actor(no, task)
                tv = ag.Q(no, na, task) - ag.alpha * ninfo['lprob']
            delta = (r + (1 - float(term)) * ag.gamma * tv - q) / sigma
            ql = (delta ** 2).sum()
            ql.backward()
            for p, g in zip(ag.Q.core.parameters(), gQ):
                g.add_(p.grad)
            pl_sum += pl.item(); ql_sum += ql.item()
            obs = next_obs
            if term or trunc:
                obs, _ = self.env.reset()
        ag.actor.zero_grad(); ag.Q.zero_grad()
        n = self.probe_steps
        return [g / n for g in gA], [g / n for g in gQ], pl_sum / n, ql_sum / n

    def _layer_cos(self, a_list, b_list):
        out = []
        for l in range(self.n_layers):
            a = _flat([t for t, li in zip(a_list, self.layer_of) if li == l])
            b = _flat([t for t, li in zip(b_list, self.layer_of) if li == l])
            out.append(_cos(a, b))
        return out

    def run(self, t):
        ag = self.agent
        gT_a, gT_q, n_upd = ag.pop_grad_track()
        gH_a, gH_q, h_pl, h_ql = self._old_task_grads()
        cur_a = [p.detach().clone() for p in ag.actor.core.parameters()]
        cur_q = [p.detach().clone() for p in ag.Q.core.parameters()]
        d_a = [c - p for c, p in zip(cur_a, self.prev_a)]
        d_q = [c - p for c, p in zip(cur_q, self.prev_q)]
        h_ret, _, h_len = evaluate_task(ag, self.old_task, self.norm_env_old, episodes=self.eval_episodes, device=ag.device)

        row = [t, n_upd, h_ret, h_len, h_pl, h_ql]
        stats = {}
        for k, gH, gT, d in (("act", gH_a, gT_a, d_a), ("crit", gH_q, gT_q, d_q)):
            fH, fT, fd = _flat(gH), _flat(gT), _flat(d)
            s = [fH.norm().item(), fT.norm().item(), _cos(fT, fH), fd.norm().item(), _cos(fd, fH), torch.dot(fH, fd).item()]
            stats[k] = s
            row += s
        row += self._layer_cos(gT_a, gH_a) + self._layer_cos(gT_q, gH_q)
        with open(self.csv_path, "a") as f:
            f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        print(f"[Probe t={t:>8}] Humanoid eval R={h_ret:8.1f} (len {h_len:5.1f}) | "
              f"cos(gT,gH) act={stats['act'][2]:+.3f} crit={stats['crit'][2]:+.3f} | "
              f"dL_H pred act={stats['act'][5]:+.4f} crit={stats['crit'][5]:+.4f}", flush=True)
        self.prev_a, self.prev_q = cur_a, cur_q


def run_stage(agent, env_name, norm_env, env, num_steps, stage_name, results_dir, run_id,
              checkpoint_freq=50000, diag_csv_path=None, anchor_core=None, print_freq=10,
              probe=None, probe_freq=0):
    print(f"\n==========================================", flush=True)
    print(f"Starting {stage_name}: {env_name} for {human_format_numbers(num_steps)} steps", flush=True)
    print(f"==========================================", flush=True)

    rets, ep_steps = [], []
    ret, step = 0, 0
    obs, _ = env.reset()
    ep_tic = time.time()
    last_diag = None

    for t in range(num_steps):
        if probe is not None and probe_freq > 0 and t % probe_freq == 0:
            probe.run(t)
        action, action_info = agent.compute_action(obs, env_name)
        sim_action = action.detach().cpu().view(-1).numpy()

        next_obs, reward, term, trunc, _ = env.step(sim_action)
        is_ep_end = bool(term or trunc)

        # Record gradient norms on episode end or every 250 steps
        rec_diag = is_ep_end or (t % 250 == 0)
        step_diag = agent.update(obs, action, next_obs, reward, term, env_name, record_diagnostics=rec_diag, **action_info)
        if step_diag is not None:
            last_diag = step_diag

        ret += reward
        step += 1
        obs = next_obs

        if t > 0 and t % checkpoint_freq == 0:
            agent.save(f"{results_dir}/{run_id}_{stage_name}_model_{human_format_numbers(t)}.pt",
                       norm_envs={env_name: norm_env})

        if is_ep_end:
            rets.append(ret)
            ep_steps.append(step)
            ep_num = len(rets)

            # Get parameter norms and drift
            param_diag = agent.get_parameter_diagnostics(env_name, anchor=anchor_core)
            d = last_diag or {}

            # Append to CSV diagnostics log
            if diag_csv_path:
                with open(diag_csv_path, "a") as f_csv:
                    f_csv.write(
                        f"{t+1},{stage_name},{env_name},{ep_num},{ret:.2f},{step},"
                        f"{d.get('a_core_gn', 0.0):.6f},{d.get('a_head_gn', 0.0):.6f},"
                        f"{d.get('q_core_gn', 0.0):.6f},{d.get('q_head_gn', 0.0):.6f},"
                        f"{param_diag['a_core_pn']:.6f},{param_diag['q_core_pn']:.6f},"
                        f"{param_diag['a_head_pn']:.6f},{param_diag['q_head_pn']:.6f},"
                        f"{param_diag['a_core_drift']:.6f},{param_diag['q_core_drift']:.6f},"
                        f"{d.get('ploss', 0.0):.6f},{d.get('qloss', 0.0):.6f},{d.get('sigma', 1.0):.6f},"
                        f"{d.get('ewc_a', 0.0):.6f},{d.get('ewc_q', 0.0):.6f},"
                        + ",".join(f"{x:.6f}" for x in param_diag['a_layer_drift']) + ","
                        + ",".join(f"{x:.6f}" for x in param_diag['q_layer_drift']) + "\n"
                    )

            if ep_num % print_freq == 0 or ret > 1000:
                drift_str = ""
                if anchor_core:
                    a_l = "/".join(f"{x:.1f}" for x in param_diag['a_layer_drift'])
                    q_l = "/".join(f"{x:.1f}" for x in param_diag['q_layer_drift'])
                    drift_str = f" | Drift:[Act={param_diag['a_core_drift']:.3f} ({a_l}),Crit={param_diag['q_core_drift']:.3f} ({q_l})]"
                    if agent.ewc is not None:
                        drift_str += f" | EWC:[Act={d.get('ewc_a', 0.0):.4f},Crit={d.get('ewc_q', 0.0):.4f}]"
                print(
                    f"[{stage_name}] E:{ep_num:5d} | S:{step:4d} | R:{ret:7.2f} | T:{t+1:7d} | "
                    f"G:[ActCore:{d.get('a_core_gn', 0.0):.3f}, ActHead:{d.get('a_head_gn', 0.0):.3f} | CritCore:{d.get('q_core_gn', 0.0):.3f}, CritHead:{d.get('q_head_gn', 0.0):.3f}] | "
                    f"P:[ActCore:{param_diag['a_core_pn']:.2f}, CritCore:{param_diag['q_core_pn']:.2f}]{drift_str}",
                    flush=True
                )
            ep_tic = time.time()
            obs, _ = env.reset()
            ret, step = 0, 0

    if probe is not None and probe_freq > 0:
        probe.run(num_steps)
    return ep_steps, rets


def main():
    parser = argparse.ArgumentParser(description="Continual Multi-Head Action Value Gradient (AVG)")
    parser.add_argument('--task1', default="Humanoid-v4", type=str)
    parser.add_argument('--task2', default="Hopper-v4", type=str)
    parser.add_argument('--stage1_steps', default=5000000, type=int, help="Steps for Stage 1 (Task 1)")
    parser.add_argument('--stage2_steps', default=1000000, type=int, help="Steps for Stage 2 (Task 2)")
    parser.add_argument('--eval_episodes', default=30, type=int, help="Number of eval episodes")
    parser.add_argument('--results_dir', default="./results_continual", type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--actor_lr', default=0.0063, type=float)
    parser.add_argument('--critic_lr', default=0.0087, type=float)
    parser.add_argument('--beta1', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha_lr', default=0.07, type=float)
    parser.add_argument('--nhid_actor', default=256, type=int)
    parser.add_argument('--nhid_critic', default=256, type=int)
    parser.add_argument('--checkpoint_freq', default=250000, type=int, help="Save model checkpoint every N steps")
    parser.add_argument('--print_freq', default=10, type=int, help="Print episode log frequency")
    # Deeper shared core + layer-wise EWC
    parser.add_argument('--n_core_layers', default=1, type=int, help="Number of shared 256x256 core layers")
    parser.add_argument('--ewc_lambdas', default="", type=str,
                        help="Comma list of EWC lambdas, one per core layer, LOWEST layer first (e.g. '10,100,1000'). "
                             "A single value is broadcast to all layers. Empty or all-zero disables EWC.")
    parser.add_argument('--fisher', default="rollout", choices=["rollout", "online", "adam"],
                        help="rollout: stream Task-1 steps after Stage 1 (A) | online: EMA during Stage 1 (B) | adam: Adam 2nd moment (C)")
    parser.add_argument('--fisher_steps', default=5000, type=int, help="Task-1 steps streamed for --fisher rollout")
    parser.add_argument('--fisher_beta', default=0.9999, type=float, help="EMA decay for --fisher online")
    parser.add_argument('--no_fisher_normalize', action='store_true', help="Use raw Fisher scale instead of mean-1 normalization")
    parser.add_argument('--load_stage1', default="", type=str, help="Skip Stage 1 and load this stage1_final.pt")
    parser.add_argument('--stage1_only', action='store_true', help="Stop after Stage 1 (save checkpoint + eval)")
    # Stage-2 gradient interference probe
    parser.add_argument('--probe_freq', default=10000, type=int, help="Run the Stage-2 interference probe every N steps (0 = off)")
    parser.add_argument('--probe_steps', default=300, type=int, help="Task-1 transitions streamed per probe to estimate its core gradient")
    parser.add_argument('--probe_eval_episodes', default=3, type=int, help="Deterministic Task-1 eval episodes per probe")
    parser.add_argument('--freeze_core', action='store_true', help="Freeze the shared cores before Stage 2 (only task-2 heads learn)")
    args = parser.parse_args()

    # Resolve layer-wise lambdas
    lam_list = [float(x) for x in args.ewc_lambdas.split(",") if x.strip()] if args.ewc_lambdas else []

    # If resuming from a Stage-1 checkpoint, its core depth wins
    stage1_state = None
    if args.load_stage1:
        stage1_state = torch.load(args.load_stage1, map_location=args.device, weights_only=False)
        args.n_core_layers = stage1_state.get("n_core_layers", 1)

    if len(lam_list) == 1:
        lam_list = lam_list * args.n_core_layers
    use_ewc = len(lam_list) > 0 and any(l > 0 for l in lam_list)
    if lam_list and len(lam_list) != args.n_core_layers:
        raise ValueError(f"--ewc_lambdas has {len(lam_list)} values but the core has {args.n_core_layers} layers")

    set_threads(args.threads)
    args.betas = [args.beta1, 0.999]
    args.device = torch.device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    tag = f"_core{args.n_core_layers}"
    if use_ewc:
        tag += f"_ewc{'-'.join(f'{l:g}' for l in lam_list)}_{args.fisher}"
    if args.freeze_core:
        tag += "_frozen"
    if stage1_state is not None:
        src_task2 = [k for k in stage1_state["task_specs"] if k != args.task1]
        tag += f"_s1from{src_task2[0].split('-')[0] if src_task2 else 'X'}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"-ContinualAVG_{args.task1}_to_{args.task2}_seed{args.seed}{tag}"
    print(f"Starting Continual Learning Run: {run_id}", flush=True)
    print(f"Device: {args.device} | CPU Threads: {torch.get_num_threads()}", flush=True)
    print(f"Shared core layers: {args.n_core_layers} | EWC: {'ON  lambdas(low->high)=' + str(lam_list) + ' fisher=' + args.fisher if use_ewc else 'OFF'}", flush=True)

    # Initialize CSV diagnostics log
    diag_csv_path = f"{args.results_dir}/{run_id}_diagnostics.csv"
    diag_live_path = f"{args.results_dir}/diagnostics_live.csv"
    layer_cols = ",".join([f"act_drift_L{l}" for l in range(args.n_core_layers)] +
                          [f"crit_drift_L{l}" for l in range(args.n_core_layers)])
    with open(diag_csv_path, "w") as f_csv:
        f_csv.write("step,stage,task,episode,return,length,act_core_grad,act_head_grad,crit_core_grad,crit_head_grad,act_core_param,crit_core_param,act_head_param,crit_head_param,act_core_drift,crit_core_drift,ploss,qloss,sigma,ewc_act,ewc_crit," + layer_cols + "\n")

    if os.path.exists(diag_live_path):
        os.remove(diag_live_path)
    try:
        os.symlink(os.path.abspath(diag_csv_path), diag_live_path)
    except Exception:
        pass

    # 1. Initialize Environments
    env1_raw = gym.make(args.task1)
    env1_flat = FlattenObservation(env1_raw)
    env1_norm = NormalizeObservation(env1_flat)
    env1 = ClipAction(env1_norm)

    env2_raw = gym.make(args.task2)
    env2_flat = FlattenObservation(env2_raw)
    env2_norm = NormalizeObservation(env2_flat)
    env2 = ClipAction(env2_norm)

    task_specs = {
        args.task1: (env1.observation_space.shape[0], env1.action_space.shape[0]),
        args.task2: (env2.observation_space.shape[0], env2.action_space.shape[0]),
    }
    print("Task Specifications:", flush=True)
    for task, (in_dim, out_dim) in task_specs.items():
        print(f"  {task:<12}: Obs Dim = {in_dim:3d}, Action Dim = {out_dim:2d}", flush=True)

    # Seed everything (previously --seed only appeared in the run name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env1.reset(seed=args.seed)
    env1.action_space.seed(args.seed)
    env2.reset(seed=args.seed + 1)
    env2.action_space.seed(args.seed + 1)

    # 2. Build Continual Multi-Head Agent
    agent = ContinualAVG(task_specs, args)

    # -------------------------------------------------------------
    # STAGE 1: Train on Task 1 (Humanoid-v4)  -- or load it
    # -------------------------------------------------------------
    if stage1_state is not None:
        if stage1_state["task_specs"] == task_specs:
            agent.actor.load_state_dict(stage1_state["actor"])
            agent.Q.load_state_dict(stage1_state["critic"])
            agent.popt.load_state_dict(stage1_state["policy_opt"])
            agent.qopt.load_state_dict(stage1_state["critic_opt"])
        else:
            old_specs = stage1_state["task_specs"]
            nh_a = stage1_state.get("nhid_actor", args.nhid_actor)
            nh_q = stage1_state.get("nhid_critic", args.nhid_critic)
            old_a = [n for n, _ in ContinualActor(old_specs, torch.device("cpu"), nh_a, args.n_core_layers).named_parameters()]
            old_q = [n for n, _ in ContinualQ(old_specs, torch.device("cpu"), nh_q, args.n_core_layers).named_parameters()]
            fresh_a = load_stage1_partial(agent.actor, agent.popt, stage1_state["actor"], stage1_state["policy_opt"], old_a)
            fresh_q = load_stage1_partial(agent.Q, agent.qopt, stage1_state["critic"], stage1_state["critic_opt"], old_q)
            print(f"Cross-task load ({list(old_specs)} -> {list(task_specs)}). Freshly initialised: actor {fresh_a} | critic {fresh_q}", flush=True)
        ns = stage1_state["norm_stats"][args.task1]
        env1_norm.obs_rms.mean, env1_norm.obs_rms.var, env1_norm.obs_rms.count = ns["mean"], ns["var"], ns["count"]
        if stage1_state.get("online_fisher") is not None:
            agent.online_fisher = stage1_state["online_fisher"]
        t1_steps, t1_rets = stage1_state.get("t1_steps", []), stage1_state.get("t1_returns", [])
        print(f"Loaded Stage 1 from: {args.load_stage1} (skipping Stage 1 training)", flush=True)
    else:
        if args.fisher == "online":
            agent.enable_online_fisher(args.task1, beta=args.fisher_beta)
        t1_steps, t1_rets = run_stage(
            agent=agent, env_name=args.task1, norm_env=env1_norm, env=env1,
            num_steps=args.stage1_steps, stage_name=f"Stage1_{args.task1.split('-')[0]}",
            results_dir=args.results_dir, run_id=run_id, checkpoint_freq=args.checkpoint_freq,
            diag_csv_path=diag_csv_path, anchor_core=None, print_freq=args.print_freq
        )
        stage1_ckpt = f"{args.results_dir}/{run_id}_stage1_final.pt"
        agent.save(stage1_ckpt, norm_envs={args.task1: env1_norm},
                   extra={"online_fisher": agent.online_fisher, "t1_returns": t1_rets, "t1_steps": t1_steps})
        print(f"Stage 1 checkpoint saved to: {stage1_ckpt}", flush=True)

    # Snapshot core weights at end of Stage 1 to monitor representation drift during Stage 2
    anchor_core = agent.snapshot_core()
    print("Core representations snapshotted for drift monitoring in Stage 2.", flush=True)

    # Evaluate Task 1 performance at end of Stage 1
    mean_ret_t1_before, std_t1_before, steps_t1_before = evaluate_task(
        agent, args.task1, env1_norm, episodes=args.eval_episodes, device=args.device
    )
    print(f"\n[Post Stage 1 Evaluation] {args.task1}: Return = {mean_ret_t1_before:.2f} +/- {std_t1_before:.2f} | Steps = {steps_t1_before:.1f}", flush=True)
    if args.stage1_only:
        print("--stage1_only set: stopping after Stage 1.")
        return

    # -------------------------------------------------------------
    # EWC setup: estimate Fisher for the shared cores (no replay buffer)
    # -------------------------------------------------------------
    if use_ewc:
        tic = time.time()
        if args.fisher == "rollout":
            F_a, F_q = fisher_from_rollout(agent, args.task1, env1, env1_norm, n_steps=args.fisher_steps)
        elif args.fisher == "online":
            res = agent.get_online_fisher()
            if res is None:
                raise RuntimeError("No online Fisher available: Stage 1 must be run (or loaded) with --fisher online")
            F_a, F_q = res
        else:
            F_a, F_q = fisher_from_adam(agent)
        agent.online_fisher = None  # stop accumulating
        if not args.no_fisher_normalize:
            F_a, F_q = normalize_fisher(F_a), normalize_fisher(F_q)
        agent.set_ewc(F_a, F_q, lam_list)
        print(f"[EWC] Fisher ({args.fisher}) computed in {time.time() - tic:.1f}s", flush=True)
        print(f"[EWC] lambdas per core layer (low->high): {lam_list}", flush=True)
        print(f"[EWC] mean Fisher per layer  actor: {[round(x, 4) for x in fisher_layer_means(agent.actor.core, F_a)]}", flush=True)
        print(f"[EWC] mean Fisher per layer critic: {[round(x, 4) for x in fisher_layer_means(agent.Q.core, F_q)]}", flush=True)

    # -------------------------------------------------------------
    # STAGE 2: Continual Transfer to Task 2 (Hopper-v4) with Shared Core
    # -------------------------------------------------------------
    probe = None
    if args.freeze_core:
        for p in list(agent.actor.core.parameters()) + list(agent.Q.core.parameters()):
            p.requires_grad_(False)
        print("[Frozen core] Shared actor & critic cores frozen for Stage 2 (only task-2 heads learn).", flush=True)
    if args.probe_freq > 0:
        agent.start_grad_track(args.task2)
        probe_csv = f"{args.results_dir}/{run_id}_stage2_probe.csv"
        probe = InterferenceProbe(agent, args.task1, env1_norm, probe_csv, probe_steps=args.probe_steps,
                                  eval_episodes=args.probe_eval_episodes, seed=args.seed + 7)
        print(f"Stage-2 interference probe every {args.probe_freq} steps -> {probe_csv}", flush=True)
    t2_steps, t2_rets = run_stage(
        agent=agent, env_name=args.task2, norm_env=env2_norm, env=env2,
        num_steps=args.stage2_steps, stage_name=f"Stage2_{args.task2.split('-')[0]}",
        results_dir=args.results_dir, run_id=run_id, checkpoint_freq=args.checkpoint_freq,
        diag_csv_path=diag_csv_path, anchor_core=anchor_core, print_freq=args.print_freq,
        probe=probe, probe_freq=args.probe_freq
    )
    agent.grad_track = None

    ewc_config = {
        "n_core_layers": args.n_core_layers,
        "ewc_lambdas": lam_list if use_ewc else [],
        "fisher": args.fisher if use_ewc else None,
        "fisher_steps": args.fisher_steps,
        "fisher_normalized": not args.no_fisher_normalize,
        "loaded_stage1": args.load_stage1 or None,
        "freeze_core": args.freeze_core,
    }
    final_diag = agent.get_parameter_diagnostics(args.task2, anchor=anchor_core)
    print(f"Final per-layer core drift (low->high)  actor: {[round(x, 2) for x in final_diag['a_layer_drift']]}")
    print(f"Final per-layer core drift (low->high) critic: {[round(x, 2) for x in final_diag['q_layer_drift']]}")

    stage2_ckpt = f"{args.results_dir}/{run_id}_stage2_final.pt"
    agent.save(stage2_ckpt, norm_envs={args.task1: env1_norm, args.task2: env2_norm}, extra={"ewc_config": ewc_config})
    print(f"Stage 2 checkpoint saved to: {stage2_ckpt}")

    # Evaluate Task 2 performance at end of Stage 2
    mean_ret_t2, std_t2, steps_t2 = evaluate_task(
        agent, args.task2, env2_norm, episodes=args.eval_episodes, device=args.device
    )
    print(f"\n[Post Stage 2 Evaluation] {args.task2}: Return = {mean_ret_t2:.2f} +/- {std_t2:.2f} | Steps = {steps_t2:.1f}")

    # -------------------------------------------------------------
    # STAGE 3: THE FORGETTING TEST (Re-evaluate Task 1 on Head 1)
    # -------------------------------------------------------------
    print("\n=======================================================")
    print(f"THE CATASTROPHIC FORGETTING TEST: Re-evaluating {args.task1}")
    print("=======================================================")
    mean_ret_t1_after, std_t1_after, steps_t1_after = evaluate_task(
        agent, args.task1, env1_norm, episodes=args.eval_episodes, device=args.device
    )
    retention_pct = (mean_ret_t1_after / max(mean_ret_t1_before, 1e-4)) * 100.0

    print(f"{args.task1} Performance Before Hopper: Return = {mean_ret_t1_before:7.2f}")
    print(f"{args.task1} Performance After  Hopper: Return = {mean_ret_t1_after:7.2f}")
    print(f"Knowledge Retention: {retention_pct:5.1f}%")

    # -------------------------------------------------------------
    # Save Full Logs & Generate Summary Plot
    # -------------------------------------------------------------
    summary_data = {
        "task1": args.task1,
        "task2": args.task2,
        "t1_returns": t1_rets,
        "t1_steps": t1_steps,
        "t2_returns": t2_rets,
        "t2_steps": t2_steps,
        "t1_eval_before": (mean_ret_t1_before, std_t1_before, steps_t1_before),
        "t2_eval": (mean_ret_t2, std_t2, steps_t2),
        "t1_eval_after": (mean_ret_t1_after, std_t1_after, steps_t1_after),
        "retention_pct": retention_pct,
        "ewc_config": ewc_config,
        "final_act_layer_drift": final_diag['a_layer_drift'],
        "final_crit_layer_drift": final_diag['q_layer_drift'],
    }
    with open(f"{args.results_dir}/{run_id}_summary.pkl", "wb") as f:
        pickle.dump(summary_data, f)

    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if len(t1_rets) > 0:
        ax1.plot(np.cumsum(t1_steps), t1_rets, color="teal", alpha=0.3, label=f"{args.task1} (Stage 1)")
        w1 = max(1, min(50, len(t1_rets)//5))
        if len(t1_rets) >= w1:
            ax1.plot(np.cumsum(t1_steps)[w1-1:], np.convolve(t1_rets, np.ones(w1)/w1, mode='valid'), color="darkcyan", linewidth=2)
    ax1.set_title(f"Stage 1: {args.task1} Training", fontweight="bold")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Return")
    ax1.grid(True, linestyle="--", alpha=0.6)

    if len(t2_rets) > 0:
        ax2.plot(np.cumsum(t2_steps), t2_rets, color="coral", alpha=0.3, label=f"{args.task2} (Stage 2)")
        w2 = max(1, min(50, len(t2_rets)//5))
        if len(t2_rets) >= w2:
            ax2.plot(np.cumsum(t2_steps)[w2-1:], np.convolve(t2_rets, np.ones(w2)/w2, mode='valid'), color="chocolate", linewidth=2)
    ax2.set_title(f"Stage 2: {args.task2} Continual Transfer", fontweight="bold")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Return")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/{run_id}_continual_curve.png", dpi=200)
    plt.close()
    print(f"\nContinual learning curve saved to: {args.results_dir}/{run_id}_continual_curve.png")
    print("Done!")


if __name__ == '__main__':
    main()
