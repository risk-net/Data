# AI 风险分析研究资源包 / AI Risk Analysis Research Resource Package

[English](#english) | [中文](#chinese)

---

## English

Welcome to the AI Risk Analysis Research Resource Package, a comprehensive collection encompassing event alignment, multi-dimensional classification of AI risk incidents, and standard datasets for AI risk research. This package provides code implementations, methodologies, and datasets to support academic research on structured analysis of AI risk incidents.

### Components

This resource package consists of three main components, each with its own detailed documentation:

1. **Event Alignment Project**Implements event alignment and clustering using vector embeddings and similarity metrics, with multiple approaches for event clustering based on different data representations.[View Documentation](Event-alignment/README.md)
2. **Multi-dimensional Classification of AI Risk Incidents**Focuses on establishing a unified taxonomy (RiskNet Taxonomy) and benchmark dataset, comparing prompt-based inference and fine-tuned LLMs in multi-dimensional classification tasks.[View Documentation](Multi-dimensional-Classification/README.md)
3. **Standard Datasets**
   Contains four standard datasets for AI risk research: standard case dataset, standard event dataset, standard event classification dataset, and manually labeled risk-related case dataset.
   [View Documentation](StandardDataset/README.md)

### Overview

#### Event Alignment Project

Provides code tools to cluster related AI risk events using text, summary, or metadata-based approaches, leveraging vector databases and multiple similarity metrics for accurate alignment—supporting reproducibility of the event clustering methodology described in the research.

#### Multi-dimensional Classification

Offers implementation frameworks for classifying AI risk incidents across multiple dimensions (entity, intent, timing, domain, and EU AI Act risk levels), along with evaluation scripts to compare model performance as detailed in the paper.

#### Standard Datasets

Supplies curated datasets that form the empirical foundation for AI risk research, including case data, aligned events, classified events, and labeled risk-related news—enabling further research and method comparisons.

### Getting Started

1. Clone the repository
2. Refer to each component's documentation for specific setup instructions:
   - [Event Alignment Setup](Event-alignment/README.md)
   - [Classification Setup](Multi-dimensional-Classification/README.md)
   - [Datasets Information](StandardDataset/README.md)

### Common Dependencies

While specific requirements vary by component, core academic research dependencies include:

- Python 3.8+
- FAISS (for vector similarity)
- scikit-learn (for evaluation metrics)
- JSONLines (for data handling)

Please check each component's documentation for detailed requirements and installation instructions.

### Citation

If you use this resource package in your research, please cite the associated paper:
[Insert citation format here once published]

### Contact

For questions related to the research implementation or datasets, please contact the corresponding author of the paper.

---

## 中文

欢迎使用 AI 风险分析研究资源包，这是一个综合集合，包含事件对齐、AI 风险事件的多维度分类以及用于 AI 风险研究的标准数据集。本资源包提供代码实现、方法论和数据集，支持对 AI 风险事件结构化分析的学术研究。

### 组件

本资源包包含三个主要组件，每个组件都有自己的详细文档：

1. **事件对齐项目**使用向量嵌入和相似性指标实现事件对齐和聚类，基于不同的数据表示提供多种事件聚类方法。[查看文档](event_alignment/README.md)
2. **AI 风险事件的多维度分类**致力于建立统一的分类体系（RiskNet 分类法）和基准数据集，比较基于提示词的推理和微调大语言模型在多维度分类任务中的表现。[查看文档](Multi-dimensional-Classification/README.md)
3. **标准数据集**
   包含四个用于 AI 风险研究的标准数据集：标准案例数据集、标准事件数据集、标准事件分类数据集和人工标注的风险相关案例数据集。
   [查看文档](datasets/README.md)

### 概述

#### 事件对齐项目

提供代码工具，通过基于文本、摘要或元数据的方法对相关 AI 风险事件进行聚类，利用向量数据库和多种相似性指标实现准确对齐，支持研究中事件聚类方法的可复现性。

#### 多维度分类

提供跨多个维度（实体、意图、时间、领域和欧盟 AI 法案风险等级）对 AI 风险事件进行分类的实现框架，以及评估脚本，用于比较论文中详述的模型性能。

#### 标准数据集

提供精心整理的数据集，构成 AI 风险研究的实证基础，包括案例数据、对齐事件、分类事件和带标签的风险相关新闻，为进一步研究和方法对比提供支持。

### 快速开始

1. 克隆仓库
2. 参考每个组件的文档获取具体的设置说明：
   - [事件对齐设置](event_alignment/README.md#环境要求)
   - [分类设置](Multi-dimensional-Classification/README.md)
   - [数据集信息](StandardDataset/README.md)

### 通用依赖

虽然各组件的具体要求有所不同，但核心的学术研究依赖项包括：

- Python 3.8+
- FAISS（用于向量相似性）
- scikit-learn（用于评估指标）
- JSONLines（用于数据处理）

请查看各组件的文档以获取详细的要求和安装说明。

### 引用

如果您在研究中使用本资源包，请引用相关论文：
[论文发表后插入引用格式]

### 联系方式

有关研究实现或数据集的问题，请联系论文的通讯作者。
