# 模型文件说明

本精简仓库不上传模型权重。运行代码时需自行准备：

- RTMDet Tiny 配置与预训练权重；
- RTMPose 配置与预训练权重；
- 学生/监考员身份与 ROI 行为分类头；
- PoseMLP 或 ROI+Pose 行为模型。

推荐本地目录：

```text
models/
├── detector/
├── rtmpose/
└── behavior/
```

模型权重通常较大，并可能受数据授权限制，因此已由 `.gitignore` 排除。
