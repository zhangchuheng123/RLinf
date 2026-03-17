高优先级问题（最可能直接打穿成功率）

绕过了 lerobot 的预处理/后处理流水线
lerobot 对 SmolVLA 明确构建了 preprocessor/postprocessor，包含 tokenizer、状态归一化、动作反归一化等步骤，见 factory.py:353 和 processor_smolvla.py。
标准评估链路也是先 preprocessor，再 policy.select_action，再 postprocessor，见 lerobot_eval.py:543。
你们 no-ray 目前是手工组 batch 后直接调 select_action，见 smolvla_policy.py:317。
这会导致输入分布、语言 token、动作尺度与 SFT 时不一致，极容易出现“看起来在跑但行为全错”。
动作时序使用方式与 SmolVLA 原生机制不一致
当前实现每步只拿一次动作，然后直接 repeat 成 num_action_chunks，见 smolvla_policy.py:318。
并且每个 rollout step 都 reset queue，见 smolvla_policy.py:312。
这基本等于“开环重复同一个动作”，在 LIBERO 任务上很容易直接失败。
模型配置里 n_action_steps 是 1（而你们执行 chunk_step），见 config.json。
模型训练目标被替换为 surrogate logprob，和原始 SmolVLA forward 脱钩
现在 logprob 不是来自 SmolVLA forward，而是用 -(action-noise)^2 代理，见 smolvla_policy.py:270。
这会让 PPO 更新信号与真实策略概率严重不一致，后续 PPO 微调很难“在 SFT 基础上提升”。
中优先级问题（会显著拖低性能）
4. 多相机输入可能不对齐

你下载的模型配置里输入是两个视觉键：observation.images.image 和 observation.images.image2，见 config.json。
当前 no-ray 默认只喂主相机，wrist 相机在配置里没启用，见 smolvla.yaml。
如果这个 SFT 模型训练时依赖双目/双视角，少一路视角会明显掉成功率。
归一化统计可能被“容错修补”掩盖真实问题
你们现在会把检测到的异常 normalizer stats 直接替换成 mean=0/std=1，见 smolvla_policy.py:106。
这能避免崩溃，但也可能把本来需要定位的数据/导出问题掩盖掉，导致策略输出尺度错位。
语言输入格式可能和 SFT 不一致
lerobot 的 SmolVLA 预处理会补换行并 tokenizer，见 processor_smolvla.py:71。
no-ray 目前直接传 task_descriptions，见 smolvla_policy.py:229。
prompt 细节不一致在 VLA 上经常会造成明显性能回退。
低优先级问题（通常不是零成功率主因）
7. 脚本强制 fp32 与 SFT 常见推理设置可能不一致

见 run_libero_ppo_smolvla_noray.sh。
一般更多是速度/数值稳定性问题，不太像“直接 0 成功率”的第一主因。
tokenizers fork 警告
这是你日志里看到的 warning，通常不是导致策略性能归零的主因，但建议在多进程前固定 TOKENIZERS_PARALLELISM。
我给你的结论是：
最该先怀疑的不是“模型不是 SFT 版”，而是“当前 no-ray 调用链没有按 lerobot 方式消费这个 SFT 模型”。

最小验证路径（建议先做，成本低）：

用同一份本地模型目录 smolvla_libero 走一遍 lerobot 标准评估链（带 pre/postprocessor），确认它在标准链路是否有非零成功率。
如果标准链路成功率正常，就说明问题在 no-ray 接入层，不在模型权重本身。
然后优先修三件事：接回 pre/postprocessor、取消 repeat 同动作 chunk、恢复真实 forward logprob。









export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false; uv run --no-sync python -m lerobot.scripts.lerobot_eval --policy.path=/home/chuheng/RLinf/models/smolvla_libero --env.type=libero --env.task=libero_10 --eval.batch_size=2 --eval.n_episodes=2 --policy.use_amp=false --policy.device=cuda
