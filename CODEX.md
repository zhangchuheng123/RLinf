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

请分析如下命令的执行过程，在 env.reset()/step() 后

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false; uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=/home/chuheng/RLinf/models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=2 --policy.use_amp=false --policy.device=cuda