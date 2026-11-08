# MiniMax-H3 FL2VA 候选源码快照

本目录整理自 2026-11-07 InfiniFabric 运行目录，包含 InfiniLM、InfiniCore、NineToothed 和 ntops 源码快照，以及参与实验的注意力构建脚本和部分预构建运行库。仓库不包含模型权重和用户图片。原平台目录不是 Git 工作副本，本仓库的提交是快照提交，不代表上游原始历史。

完整视频实验使用了事先构建的诊断版 InfiniLM 扩展和 NineToothed 注意力共享库，见https://github.com/cheems12/ninetooth_adaptive_lala/tree/main/artifacts/hygon-gfx936
由于算力平台时长限制，从该快照干净编译并再次完整生成尚未验证.

统一检查和运行入口为 `run_fl2va.sh`，所需环境、使用步骤和已知限制见 `docs/构建与运行.md`。

队伍：啦啦队；成员：赵玮晨、王昱霖、杜傲源。比较记录与阶段报告见 `docs/`。
