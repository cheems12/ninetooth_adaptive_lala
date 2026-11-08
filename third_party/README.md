# 离线依赖快照

本目录包含实验环境使用过的部分第三方 Python wheel、diffusers 源码归档和 torchvision 运行组件；`torchvision-LICENSE.txt` 为其源码许可证。原始 wheel 中的许可证和元数据随归档保留；各组件的版本与许可证以包内元数据为准。它们只覆盖一部分运行依赖，不能代替海光平台的 PyTorch、DTK/HIP 或完整的设备运行时。不要用公共 PyPI PyTorch 覆盖平台版本。

本目录没有包含完整的海光版 Triton 和 FlashAttention 软件包。它们可能需要由比赛环境或兼容的离线依赖目录提供；第三方组件在正式推理路径中的使用范围和比赛合规性仍需确认。
