<h1 align="center">MLMD机器学习元数据管理系统</h1>
<p align="center">
  <a href="README.md"><strong>English</strong></a> | <strong>简体中文</strong>
</p>



## 目录

- [仓库简介](#项目介绍)
- [前置条件](#前置条件)
- [镜像说明](#镜像说明)
- [获取帮助](#获取帮助)
- [如何贡献](#如何贡献)

## 项目介绍

[MLMD](https://github.com/tensorflow/metadata) 是由Google推出的开源元数据管理组件，专为机器学习系统设计，记录并管理机器学习模型训练、数据集等各种信息，能够实现模型的可追溯性、可重复性和合规性审查。本商品基于鲲鹏服务器的Huawei Cloud EulerOS 2.0 64bit系统，提供开箱即用的MLMD。

## 核心特性

- **端到端可视化机器学习流水线：** 支持用户通过 Web 界面上传数据集、自动完成数据预处理、模型训练、评估与部署，实现“零代码”建模体验，降低 ML 使用门槛
- **全链路元数据追踪与可追溯性：** 基于 MLMD 架构自动记录数据集版本、模型参数、训练过程及输出结果，构建从数据到模型的完整血缘图谱，支持审计、复现与问题溯源

本项目提供的开源镜像商品 [MLMD机器学习元数据管理系统](https://marketplace.huaweicloud.com/hidden/contents/10e68a87-10f0-4245-a3f4-77eea4e91916#productid=OFFI1148940556404793344) 已预先安装1.14.0版本的MLMD及其相关运行环境，并提供部署模板。快来参照使用指南，轻松开启“开箱即用”的高效体验吧。

> **系统要求如下：**
>
> - CPU: 2vCPUs 或更高
> - RAM: 4GB 或更大
> - Disk: 至少 40GB

## 前置条件

[注册华为账号并开通华为云](https://support.huaweicloud.com/usermanual-account/account_id_001.html)

## 镜像说明

| 镜像规格                                                     | 特性说明                                                 | 备注 |
| ------------------------------------------------------------ | -------------------------------------------------------- | ---- |
| [MLMD-1.14.0-kunpeng](https://github.com/HuaweiCloudDeveloper/mlmd-image/tree/MLMD-1.14.0-kunpeng) | 基于鲲鹏服务器 + Huawei Cloud EulerOS 2.0 64bit 安装部署 |      |

## 获取帮助

- 更多问题可通过 [issue](https://github.com/HuaweiCloudDeveloper/mlmd-image/issues) 或 华为云云商店指定商品的服务支持 与我们取得联系
- 其他开源镜像可看 [open-source-image-repos](https://github.com/HuaweiCloudDeveloper/open-source-image-repos)

## 如何贡献

- Fork 此存储库并提交合并请求
- 基于您的开源镜像信息同步更新 README.md
