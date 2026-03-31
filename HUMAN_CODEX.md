# Q1

分析如下代码产生的效果和作用，如何测试改模型完成任务的准确率呢？先不要改动代码

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false; uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=/home/chuheng/RLinf/models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=2 --policy.use_amp=false --policy.device=cuda

# Q2

examples/embodiment/run_libero_ppo_smolvla_noray.sh 本来是需要基于一个 SFT 之后的 SmolVLA 来进行 PPO 微调的，但是现在的问题是 SFT 之后的 SmolVLA 成功率为零，没办法在其基础上微调。现在怀疑是我们使用微调之后的 SmolVLA 模型方式不正确（比如归一化，输入输出对齐等）导致的。我们使用的 SFT SmolVLA 的推荐使用方法是用 lerobot，相应的代码已经拷贝到 lerobot/ 下了，调用方式如下 
lerobot-record \
  --robot.type=so100_follower \
  --dataset.repo_id=<hf_user>/eval_<dataset> \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --episodes=10。
Huggingface 上的 HuggingFaceVLA/smolvla_libero 已经下载到了 models/ 下。
根据初步观察，发现 lerobot 链路模型表现正常，但是 rlinf_noray 链路下模型不正确，请比较两者差异。

# Q3

请调整 rlinf_noray 链路使其对齐 lerobot 链路。
1）复制 lerobot 的预处理/后处理链路，替代原来的手搓链路；
2）action_chunk 等时序问题也和 lerobot 对齐；
3）保留 PPO logprob 相关训练逻辑；
4）启用双相机，和 lerobot 对齐；
5）着重检查和对齐归一化；

注意：
1）后续 lerobot 文件夹和 rlinf 文件夹都会弃用，请不要依赖它们；主要逻辑都在 rlinf_noray 中完成；
2）如果需要修改相关的配置文件，也请一并修改；
3）目标是执行如下命令后可以完成配置和运行：【配置】bash install.sh embodied --model smolvla --env maniskill_libero 【运行】bash examples/embodiment/run_libero_ppo_smolvla_noray.sh
4）请用尽量少的修改完成任务，不要通过设置默认值和各种兜底来掩盖预期外的错误，有错误请让它及时抛出
5）请对于新增代码写单元测试，分模块调试，确认正确实现。

# Q4

lerobot 中的 rl 模块应该如何使用，结合分析代码和搜索相关文档回答。

# Q5

以 bash examples/embodiment/run_libero_ppo_smolvla_noray.sh 为主入口，请分析该算法的主要逻辑，指出值得重点关注的核心代码。

# Q6

请调试 bash examples/embodiment/run_libero_ppo_smolvla_noray.sh。

目标：确保其还未进行 PPO，只是第一轮 rollout 时，能有一定成功率。这说明，模型推理链路和 lerobot 已经对齐。已知，/home/chuheng/RLinf/models/smolvla_libero 模型在 libero_10 上成功率大约为 40%。

建议做法：在 rollout 函数中插入打印成功率的代码，在 rollout 第一轮后就退出程序；如果 rollout 数据量不够，可以直接在代码中增加数据量。

注意：
- 临时增加的代码套在注释 ####### TEMP START/END ####### 中；
- 请用尽量少的修改完成任务，不要通过设置默认值和各种兜底来掩盖预期外的错误，有错误请让它及时抛出；
- 调试中请禁用 wandb

# Q7

查看前面的视频，发现确实机械臂在移动，但是并不能很好地完成任务。怀疑还是目前的 rollout 和 lerobot 中的 rollout 还是没有对齐，导致任务执行失败。请做这样的操作来核对 rlinf_noray 和 lerobot 中 rollout 的核心差距，现在只核对单步运行的输入输出。

1）由于策略中会有 noise 采样，请小改策略，使其能够支持使用外部的 noise 采样，从而使得策略在给定 noise 下确定性；
2）在 lerobot 的环境中采集到的 obs、instruction、proprio 等信息打包，然后也把 a）预处理后的 payload；b）策略输出的动作；c）输入给环境的动作 等都打包，方便后续比对。
3）在 rlinf_noray 环境中在单步中读取前面的包，然后运行单步推理，观察是否每一步能对得上。通过哪一步对不上，来 debug。

注意：
- 临时增加的代码套在注释 ####### TEMP START/END ####### 中，和 bug 修复相关的代码改动不套在该注释中；
- 请用尽量少的修改完成任务，不要通过设置默认值和各种兜底来掩盖预期外的错误，有错误请让它及时抛出；
- 调试中请禁用 wandb

# Q8

目前观察到，即使都是从 models/smolvla_libero 中载入的模型，但是他们在对于同样的输入做 _get_action_chunk 时，得到的输入仍然不一样。现在的目标就是希望能找到其中的原因。

现在的运行逻辑是：
1）运行 lerobot 链路来得到 dump 后的各种信息
PYTHONPATH=. RLINF_SELECT_ACTION_ALIGN_DUMP_PATH=logs/rollout_alignment/select_action_align.pt MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false uv run --no-sync python lerobot/scripts/lerobot_eval.py --policy.path=models/smolvla_libero --env.type=libero --eval.batch_size=2 --eval.n_episodes=2 --policy.device=cuda --policy.use_amp=false
2）运行 noray 链路以对比
PYTHONPATH=. RLINF_SELECT_ACTION_ALIGN_DUMP_PATH=logs/rollout_alignment/select_action_align.pt MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false bash examples/embodiment/run_libero_ppo_smolvla_noray.sh

请保持这个工作流程，来验证和排除可能的问题：
1）强制统一 eval 和 backend；
2）对比指纹模型；
3）对齐计算精度；

注意：
- 目前运行的机器只有一块 GPU，所以 NPROC_PER_NODE 已经是 1 了，不用考虑这个问题；
- 目前代码整体处于调试状态，请不要关注无关部分的临时代码，也不要对他们做修改；
- 临时增加的代码套在注释 ####### TEMP START/END ####### 中，和 bug 修复相关的代码改动不套在该注释中；
- 请用尽量少的修改完成任务，不要通过设置默认值和各种兜底来掩盖预期外的错误，有错误请让它及时抛出；
- 调试中请禁用 wandb


# Q9

刚刚发现如果把 lerobot 和 noray 链路都换成 float32 时，两边的输出能对上。现在需要修改代码，对比 lerobot-record 下，float32 和 bf16 下的成功率区别。原本的执行命令如下。1）请增加运算精度选项，2）并且分别运行后，统计任务成功率，3）对于 10 个任务的成功率，请分别打印出来，每行为 task_x, instruction="xx", success=[T,F,...]。

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false; uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=/home/chuheng/RLinf/models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=2 --policy.use_amp=false --policy.device=cuda


# Q10

回到主线任务，以 bash examples/embodiment/run_libero_ppo_smolvla_noray.sh 为主入口，把模型推理和训练放在 bf16 下完成。请直接修改 bash 中的默认选项，并检查和修改后续程序以支持 bf16 运行。程序中有一些 TEMP 以及 dump 验证的代码，请暂时不要删掉他们，将它们注释（后续删除）。同时还有一个地方使用了 _get_action_chunk 作为临时的动作比对，请把它换成正确的函数。

注意：
- 目前运行的机器只有一块 GPU；
- 请用尽量少的修改完成任务，不要通过设置默认值和各种兜底来掩盖预期外的错误，有错误请让它及时抛出；
- 调试中请禁用 wandb

# 评估

fp32
env -u RLINF_SELECT_ACTION_ALIGN_DUMP_PATH -u LEROBOT_EVAL_DUMP_PATH PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=2 --policy.use_amp=false --policy.device=cuda --inference_precision=fp32 | tee logs/eval_fp32.log

Per-task success details:
task_0, instruction="", success=['F', 'F']
task_1, instruction="", success=['F', 'F']
task_2, instruction="", success=['F', 'F']
task_3, instruction="", success=['F', 'F']
task_4, instruction="", success=['F', 'F']
task_5, instruction="", success=['F', 'T']
task_6, instruction="", success=['F', 'F']
task_7, instruction="", success=['F', 'F']
task_8, instruction="", success=['F', 'F']
task_9, instruction="", success=['T', 'F']

bf16
env -u RLINF_SELECT_ACTION_ALIGN_DUMP_PATH -u LEROBOT_EVAL_DUMP_PATH PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=2 --policy.use_amp=false --policy.device=cuda --inference_precision=bf16 | tee logs/eval_bf16.log

Per-task success details:
task_0, instruction="", success=['F', 'F']
task_1, instruction="", success=['T', 'F']
task_2, instruction="", success=['T', 'T']
task_3, instruction="", success=['F', 'T']
task_4, instruction="", success=['F', 'F']
task_5, instruction="", success=['T', 'F']
task_6, instruction="", success=['F', 'T']
task_7, instruction="", success=['F', 'F']
task_8, instruction="", success=['F', 'F']
task_9, instruction="", success=['F', 'F']

提取 10 个任务成功率行：

grep '^task_' logs/eval_fp32.log
grep '^task_' logs/eval_bf16.log

env -u RLINF_SELECT_ACTION_ALIGN_DUMP_PATH -u LEROBOT_EVAL_DUMP_PATH PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=10 --policy.use_amp=false --policy.device=cuda --inference_precision=bf16 2>&1 | tee -a logs/eval_bf16_long.log

# Q11

请在 rlinf_noray/runners/libero_ppo_ddp_runner.py 代码中针对 collect_samples 函数运行速度慢的问题，查找原因。具体方法是，额外写一份测试代码，复现该函数的执行过程，如果有不清楚的参数配置，请参考以 bash examples/embodiment/run_libero_ppo_smolvla_noray.sh 为主入口时，相应的参数。请运行该测试代码，然后总结该函数运行慢的原因，并提出改进意见。特别地，我们关注模型推理和环境单步的耗时，并且关心并行如何影响耗时，是否有一个好的并行方案有效提速。

要求：
- 不要侵入原有代码；
- 直接从 bash 执行中推测各个参数数值，然后硬编码到测试代码中，要求测试代码简洁。


# Q12

目前需要比较如下两个链路中的模型表现，并尝试对齐：

A）export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false; uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=/home/chuheng/RLinf/models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=2 --policy.use_amp=false --policy.device=cuda
B）_collect_rollouts 部分： DISABLE_WANDB=1 SAVE_ROLLOUT_VIDEO=true bash examples/embodiment/run_libero_ppo_smolvla_noray.sh

要求：

1）把这两个链路都改成在 env.reset()/step() 后进行 noise sample shape=(bsize, self.config.chunk_size, self.config.max_action_dim)，然后传递这个 noise 进 policy 内部，进行动作采样，从而在 condiiton on noise 的情况下消除随机性。
2）在 A 链路中的 env.reset()/step() 后把 obs 等保存为 pickle；在 B 链路中读出该 pickle。后续对比的事情我来做。


# Q13

RLINF_ROLLOUT_ALIGN_PICKLE_PATH="/home/chuheng/RLinf/logs/q12_align/align.pkl" PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=2 --policy.use_amp=false --policy.device=cuda --inference_precision=bf16 

RLINF_ROLLOUT_ALIGN_PICKLE_PATH="/home/chuheng/RLinf/logs/q12_align/align.pkl" DISABLE_WANDB=1 SAVE_ROLLOUT_VIDEO=true bash examples/embodiment/run_libero_ppo_smolvla_noray.sh

# Q14

请调整 noray 代码，对齐如下两个链路：

1）noray 链路：
- rlinf_noray/envs/libero/libero_env_lerobot_adapter.py : overwrite observation = align_data["observation_from_env"] from dumped data
- rlinf_noray/envs/libero/libero_env_lerobot_adapter.py predict_action_batch 
- rlinf_noray/models/embodiment/smolvla/smolvla_policy_lerobot_aligned.py _preprocess_obs_batch

2) lerobot 链路：
- lerobot/scripts/lerobot_eval.py : after env.reset() dumps observation_from_env 
- preprocess_observation
- add_envs_task
- env_preprocessor
- preprocessor
- observation.state related

使得，如下命令中的 assert torch.all_close(batch_obs_align['observation.state'], batch_obs['observation.state']) 检查能通过。

RLINF_ROLLOUT_ALIGN_PICKLE_PATH="/home/chuheng/RLinf/logs/q12_align/align.pkl" DISABLE_WANDB=1 SAVE_ROLLOUT_VIDEO=true bash examples/embodiment/run_libero_ppo_smolvla_noray.sh

# Q15

我们现在需要来考虑 reset 相关的问题，目前 lerobot/envs/libero.py 中的 step 和 .venv/lib/python3.11/site-packages/gymnasium/vector/sync_vector_env.py 中的 step 都包含了 autoreset 逻辑。但是这其实不是我们想要的。我们希望的表现是，在 rlinf_noray/envs/libero/libero_env_lerobot_adapter.py 中的 chunk_step 过程中，如果遇到了 done := (terminated or truncated)，chunk 内的后续步不要再 step 了，append 各种空常量，等到 chunk 完全结束之后，对相应的环境进行 reset，并且把 reset 后的 obs 放在最后的 obs 中（这样后续的策略决策，可以根据这个 obs，在下一个新 episode 上做决策）。要实现这样的做法，应该做什么改动。注意，尽量只改动 rlinf_noray 中的代码，可以考虑少量改动 lerobot 上的代码，不要更高 .venv 上的代码。请先更有条理地复述我这里讲的逻辑，然后给出一个修改方案，暂时先不修改。

注意：
- 请用尽量少的修改完成任务，不要通过设置默认值和各种兜底来掩盖预期外的错误，有错误请让它及时抛出；
- 调试中请禁用 wandb

请这样实现：1）在 chunk_step 中调用 self.env.envs[i] 来手动 step，来绕过 SyncVectorEnv 中的 step，不过请注意对齐原本的行为；2）不用去除 LiberoEnv 中的 autoreset 逻辑， step 中 done 之后会自动 reset，这并没有问题，只不过需要把 observation 返回；3）接下来可以在 chunk_step 中实现前面描述的逻辑了。请按照这个思路修改代码。

# Q16

请创建新的 examples/embodiment/run_libero_dsrl_smolvla_noray.sh 脚本以及相应的代码文件，用以实现 DSRL（具体信息见下）。

注意：
- 请尽量在现在 ppo 的代码基础上进行，尽量遵循原框架。为了避免损害已有代码，可以把代码进行复制后再更改；
- 请用尽量少的修改完成任务，不要通过设置默认值和各种兜底来掩盖预期外的错误，有错误请让它及时抛出；
- 调试中请禁用 wandb

```markdown
# DSRL-Style 实现要点 (SMOLVLA + RLINF)

## 1. 核心思想

- 不优化原始 action space，而是在 latent noise space w 训练策略 π^w(s)
- 预训练的 SmolVLA (π_θ) 完全冻结，作为黑盒生成器
- 给定 (s, w)，π_θ 生成完整 action chunk A = [a_1, ..., a_T]
- 实际只执行前 K 个动作 A[:K]

## 2. 网络架构

- pi_w (Actor): 输入观测 s，输出高斯分布 (μ, σ) 的噪声空间策略
- Q_w (Critic): 输入 (s, w)，输出 Q 值
- V_w (Value): 输入 s，输出状态价值（用于 GAE）

以上的 s 输入可以复用 SmolVLA ，提取其中的 hidden 作为网络输入，比如用最后一层最后一个 token 的 hidden

## 3. 数据采集

- w ~ π_w(s)
- A = π_θ(s, w) 生成完整 chunk
- A_exec = A[:K] 执行
- 存储 (s, w, A, r, s') 到 replay buffer

虽然只执行 K 步，但是也算做整个 chunk 的效果。

## 4. PPO 更新步骤

1. 从 buffer 采样 batch
2. 计算旧策略 log_prob(w|s)
3. 计算 GAE 优势估计 (需要 V_w)
4. 计算新策略 log_prob(w|s)
5. PPO clipped objective: L = -min(ratio * advantage, clamp(ratio) * advantage)
6. 梯度更新 + 熵正则化

## 5. Q 函数更新

- q_pred = Q_w(s, w)
- q_target = r + γ * Q_w_target(s', π_w(s').sample())
- L_Q = MSE(q_pred, q_target)
- 软更新目标网络

## 6. Action Chunk 处理

- T = 完整 chunk 长度 (如 16)
- K = 实际执行长度 (如 4)
- 每步重新生成，不复用未执行的动作
- 奖励 r 来自执行的 K 个动作

## 7. 关键公式

- 噪声策略 log_prob: log N(w|μ,σ²) = -0.5 * ||w-μ||²/σ² - d/2*log(2πσ²)
- GAE: advantage_t = δ_t + γλ * advantage_{t+1}
- PPO ratio: exp(log_prob_new - log_prob_old)

## 8. 训练稳定性

- 梯度裁剪 (max_norm=0.5)
- 熵正则化 (coef=0.01)
- 观测归一化
```

# Q18

基础信息：
- rlinf 是作为参考的目录，不要引用和修改其中的内容；rlinf_noray（简称 noray）是主要的工作目录；lerobot 是辅助目录，可以可以修改其中内容，但尽量少改。
- 请用尽量少的修改完成任务，不要通过设置默认值和各种兜底来掩盖预期外的错误，有错误请让它及时抛出；比如，不要使用 get('xxx', default)。
- 程序主入口为 SAVE_ROLLOUT_VIDEO=1 bash examples/embodiment/run_libero_dsrl_smolvla_noray.sh，主逻辑在 rlinf_noray/runners/libero_dsrl_ddp_runner.py 中。

任务要求：
1）维护一个 replay buffer（一个新的类，直接写在 rlinf_noray/runners/libero_dsrl_ddp_runner.py 中），它最多能装最近的 3000 个 transitions。assert 该参数要求能被训练并行 env 数量整除。内部维护 data = list of transitions, 每个并行 env 有一个 transitions 队列。

2）在 ppo_update 中，每次对于 minibatch_size = 1024 个 transition 采样（有放回），然后更新 update_epoch=10 次。这里也调用 replay buffer 的采样功能，采样之前，buffer 里面的量应该都计算完毕了。value net 的学习率需要是 policy net 的两倍：policy = 1e-5, value = 2e-5。

3）在 mode=train 时 collect rollout 的时候，除了最开始需要 reset 之外，其他时候都需要接着上一次继续 rollout。但是 mode=eval 的时候行为不变。rollout 返回的结果应该为 list of transitions, 然后通过 replay buffer 的方法添加进去。

4）replay buffer 添加新数据之后，通过 done 来观察轨迹是否完整。如果完整，则对其向前计算 Monte Carlo returns；否则，不对其计算 returns。只有完整的轨迹（标记为有效）可以参与到后续的 PPO 采样中。replay buffer 通过 done=True 那个 transition 上的 reward 是否大于零来判断这条轨迹是否成功，然后保存最近 20 条轨迹的成功或者失败信息，这一步上输出成功轨迹条数、总轨迹条数（最多20）、成功率。

5）在 PPO update 中，给定特定 value model 之后，再利用 replay buffer 自身的方法，对其中所有有效的 transition 预测 value。注意，现在代码重复计算了 values 和 next_values，这其实低效了，不要这样做。然后根据新标的 values，再往回捋，计算 GAE。这里注意实现要高效，可以按照 num_envs 并行化；非有效的数据按照 done 一样处理，不会往回传。跨轨迹的数据因为有 done 分割，也不会把后一条的信息传回前一条。这里请格外注意计算的正确性。计算完 advantages 之后还需要对 advantage 做归一化。

注意：
- buffer 中每个 transition 大概有这些量 (state, action, reward, next_state, done, effective, returns, advantage)，如果有没考虑全的，可以再增加。
- 上面提到的均为默认参数，请把它们写到配置文件中。

参考：GAE 计算

for i in reversed(range(batch_size)):
    returns[i] = rewards[i] + self.config.gamma * prev_return * masks[i]
    deltas[i] = rewards[i] + self.config.gamma * prev_value * masks[i] - values[i]
    advantages[i] = deltas[i] + self.config.gamma * self.config.lamda * prev_advantage * masks[i]

    prev_return = returns[i]
    prev_value = values[i]
    prev_advantage = advantages[i]

# Q19

1）修复使用 MC returns 做训练，保留 value_target 的计算，但是最后学习的时候使用 returns；2）监测 update_epoch 第一轮时的 ratio，该 ratio 应该接近 1（当 replay buffer 不复用上一轮样本的时候应该为 1）；3）在配置中先关闭 entropy loss。列出更为详细一点的改动计划，然后进行改动。

# Q20

请先帮我写一份 AGENTS.md 文档，用于给 agent 来读取，方便它了解本工作目录的主要使用方法。包括，1）本项目的主要目标：在 LIBERO 环境上调试成功强化学习算法，实现一个样本效率和计算效率都比较高的强化学习算法；2）本项目的工作目录结构：主要是 lerobot 会作为参考和引用，rlinf/ 是原始目录，仅作参考；rlinf_noray/ 是主要的工作目录，程序的主入口目前是 examples/embodiment/run_libero_dsrl_smolvla_noray.sh，环境的配置方案是 requirements/setup.sh；3）本项目目前的状态：需要把 DSRL 调通，在单一任务上能做到 100% 的准确率。

# Q21

请帮我继续 debug 和调参这个 DSRL 的代码，程序主入口是 examples/embodiment/run_libero_dsrl_smolvla_noray.sh。现在的症状是 value_loss 似乎没有特别地下降，success_rate 上涨了一些，但是稳定在 40% 的准确率学不动了。