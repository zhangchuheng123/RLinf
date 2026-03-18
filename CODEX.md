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



<class 'lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy'>
lerobot.policies.smolvla.modeling_smolvla
/home/chuheng/RLinf/lerobot/policies/smolvla/modeling_smolvla.py
321

<class 'lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy'>
lerobot.policies.smolvla.modeling_smolvla
/home/chuheng/RLinf/lerobot/policies/smolvla/modeling_smolvla.py
321

(Pdb) observation.keys()
dict_keys(['action', 'next.reward', 'next.done', 'next.truncated', 'info', 'task', 'observation.images.image', 'observation.images.image2', 'observation.state', 'observation.language.tokens', 'observation.language.attention_mask'])

(Pdb) batch_obs_for_select.keys()
dict_keys(['action', 'next.reward', 'next.done', 'next.truncated', 'info', 'task', 'observation.images.image', 'observation.images.image2', 'observation.state', 'observation.language.tokens', 'observation.language.attention_mask'])

(Pdb) temp_policy_noise
tensor([[[ 0.9665,  0.0511, -0.6645,  ...,  0.4024, -0.7788,  0.4171],
         [-0.2598,  0.0815, -1.0142,  ...,  0.0211,  0.3413,  0.0131],
         [-0.8441,  0.8453,  1.8891,  ..., -0.1589, -0.4959, -0.0930],
         ...,
         [-0.3056,  0.3587, -1.9614,  ...,  0.9520, -0.8076,  0.5202],
         [ 2.2708,  1.8357, -0.1745,  ..., -0.2052,  0.3127,  1.1329],
         [ 1.0575,  0.2723,  0.7698,  ...,  2.7843, -1.1662, -0.5889]],

        [[-0.2819, -0.5162, -1.0626,  ..., -1.0992,  0.0857, -3.2850],
         [ 0.7463,  1.0874, -0.2533,  ..., -2.2924,  0.0428,  0.4297],
         [-1.1244,  0.5943, -0.9901,  ...,  0.9637, -0.9362,  1.0462],
         ...,
         [-0.2182,  1.8499,  0.8909,  ...,  0.0063, -1.8310, -0.2732],
         [-0.8894,  0.4091, -0.2138,  ...,  1.1741,  0.5451, -1.5335],
         [ 0.2661, -1.1529, -1.3710,  ...,  0.2762, -0.3133, -1.6084]]],
       device='cuda:0')

(Pdb) step_noise_for_select
tensor([[[ 0.9665,  0.0511, -0.6645,  ...,  0.4024, -0.7788,  0.4171],
         [-0.2598,  0.0815, -1.0142,  ...,  0.0211,  0.3413,  0.0131],
         [-0.8441,  0.8453,  1.8891,  ..., -0.1589, -0.4959, -0.0930],
         ...,
         [-0.3056,  0.3587, -1.9614,  ...,  0.9520, -0.8076,  0.5202],
         [ 2.2708,  1.8357, -0.1745,  ..., -0.2052,  0.3127,  1.1329],
         [ 1.0575,  0.2723,  0.7698,  ...,  2.7843, -1.1662, -0.5889]],

        [[-0.2819, -0.5162, -1.0626,  ..., -1.0992,  0.0857, -3.2850],
         [ 0.7463,  1.0874, -0.2533,  ..., -2.2924,  0.0428,  0.4297],
         [-1.1244,  0.5943, -0.9901,  ...,  0.9637, -0.9362,  1.0462],
         ...,
         [-0.2182,  1.8499,  0.8909,  ...,  0.0063, -1.8310, -0.2732],
         [-0.8894,  0.4091, -0.2138,  ...,  1.1741,  0.5451, -1.5335],
         [ 0.2661, -1.1529, -1.3710,  ...,  0.2762, -0.3133, -1.6084]]],
       device='cuda:0')

(Pdb) temp_policy_noise.shape
torch.Size([2, 50, 32])


SmolVLAPolicy(
  (model): VLAFlowMatching(
    (vlm_with_expert): SmolVLMWithExpertModel(
      (vlm): SmolVLMForConditionalGeneration(
        (model): SmolVLMModel(
          (vision_model): SmolVLMVisionTransformer(
            (embeddings): SmolVLMVisionEmbeddings(
              (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), padding=valid)
              (position_embedding): Embedding(1024, 768)
            )
            (post_layernorm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          )
        )
        (lm_head): Linear(in_features=960, out_features=49280, bias=False)
      )
      (lm_expert): LlamaModel(
        (embed_tokens): None
        (layers): ModuleList(
        )
        (norm): LlamaRMSNorm((480,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
    )
    (state_proj): Linear(in_features=32, out_features=960, bias=True)
    (action_in_proj): Linear(in_features=32, out_features=480, bias=True)
    (action_out_proj): Linear(in_features=480, out_features=32, bias=True)
    (action_time_mlp_in): Linear(in_features=960, out_features=480, bias=True)
    (action_time_mlp_out): Linear(in_features=480, out_features=480, bias=True)
  )
)


(Pdb) self.policy.model.action_out_proj.weight
Parameter containing:
tensor([[-0.0391, -0.0235, -0.0213,  ...,  0.0351,  0.0236, -0.0221],
        [-0.0375, -0.0331, -0.0077,  ..., -0.0181, -0.0024,  0.0031],
        [-0.0331, -0.0285,  0.0114,  ...,  0.0045, -0.0217,  0.0056],
        ...,
        [ 0.0022,  0.0285, -0.0311,  ..., -0.0355, -0.0003,  0.0088],
        [-0.0334, -0.0151,  0.0280,  ...,  0.0326, -0.0443,  0.0325],
        [-0.0238, -0.0227, -0.0024,  ..., -0.0235,  0.0236,  0.0444]],
       device='cuda:0', requires_grad=True)

(Pdb) policy.model.action_out_proj.weight
Parameter containing:
tensor([[-0.0391, -0.0235, -0.0213,  ...,  0.0351,  0.0236, -0.0221],
        [-0.0375, -0.0331, -0.0077,  ..., -0.0181, -0.0024,  0.0031],
        [-0.0331, -0.0285,  0.0114,  ...,  0.0045, -0.0217,  0.0056],
        ...,
        [ 0.0022,  0.0285, -0.0311,  ..., -0.0355, -0.0003,  0.0088],
        [-0.0334, -0.0151,  0.0280,  ...,  0.0326, -0.0443,  0.0325],
        [-0.0238, -0.0227, -0.0024,  ..., -0.0235,  0.0236,  0.0444]],
       device='cuda:0', requires_grad=True)



lerobot
(Pdb) action
tensor([[ 0.1549, -0.4528,  0.1119, -0.0363,  0.1589, -0.1289, -0.9469],
        [-0.1053, -0.7577, -0.0380,  1.1149,  0.2289, -0.1882, -0.9546]],
       device='cuda:0')

noray
(Pdb) action_norm_step
tensor([[ 0.1042, -0.3413,  0.1024, -0.0116,  0.1463, -0.1367, -0.9322],
        [-0.2476, -0.9355, -0.1695,  1.6783,  0.2895, -0.3067, -0.9436]],
       device='cuda:0')