# Event Alignment Project / 事件对齐项目

[English](#english) | [中文](#chinese)

---

## English

This project implements event alignment and clustering using vector embeddings and similarity metrics. It provides multiple approaches for event clustering based on different data representations.

### Project Structure

```
event_alignment/
├── accuracy.py                    # Evaluation metrics for clustering results
├── event_alignment_text.py        # Text-based event alignment
├── event_alignment_summary.py     # Summary-based event alignment  
├── event_alignment_metadata.py    # Metadata-based event alignment
├── data/                          # Data files
│   ├── dataset.jsonl             # Main dataset
│   ├── dataset_summary.jsonl     # Summarized dataset
│   └── new_incidents.jsonl       # Ground truth clustering
├── event_alignment_text/          # Output directory for text alignment
├── event_alignment_summary/       # Output directory for summary alignment
└── event_alignment_metadata/      # Output directory for metadata alignment
```

### Features

- **Multiple Alignment Approaches**: Three different methods for event clustering
  - Text-based: Uses full text content for vectorization
  - Summary-based: Uses summarized content for vectorization  
  - Metadata-based: Uses metadata fields for vectorization

- **Vector Database**: FAISS-based vector storage and similarity search

- **Comprehensive Similarity Metrics**: 
  - FAISS vector similarity
  - Subject similarity (exact/fuzzy matching)
  - Location similarity
  - Tags similarity (Jaccard)
  - Date similarity

- **Evaluation Metrics**: 
  - Cluster accuracy
  - Macro accuracy
  - Normalized Mutual Information (NMI)

### Requirements

```bash
pip install faiss-cpu numpy tqdm xinference jsonlines scikit-learn scipy
```

### Usage

#### 1. Text-based Event Alignment

```bash
python event_alignment_text.py
```

This approach:
- Uses full text content for vectorization
- Weight configuration: FAISS=1.0, others=0.0
- High similarity threshold: 0.78

#### 2. Summary-based Event Alignment

```bash
python event_alignment_summary.py
```

This approach:
- Uses summarized content for vectorization
- Weight configuration: FAISS=0.6, others=0.1 each
- High similarity threshold: 0.75

#### 3. Metadata-based Event Alignment

```bash
python event_alignment_metadata.py
```

This approach:
- Uses metadata fields for vectorization
- Weight configuration: FAISS=0.6, others=0.1 each
- High similarity threshold: 0.6

#### 4. Evaluate Results

```bash
python accuracy.py
```

This will evaluate clustering results against ground truth and output:
- Cluster accuracy
- Macro accuracy  
- Normalized Mutual Information (NMI)

### Configuration

#### Vectorizer Settings

Update the Xinference URL and model name in each alignment file:

```python
vectorizer = TextVectorizer(
    xinference_url="your_xinference_url",
    model_name="bge-m3"  # or other embedding model
)
```

#### Similarity Weights

Adjust weights in each alignment file:

```python
WEIGHT_FAISS = 0.6      # FAISS vector similarity weight
WEIGHT_SUBJECT = 0.1    # Subject similarity weight
WEIGHT_LOCATION = 0.1   # Location similarity weight
WEIGHT_TAGS = 0.1       # Tags similarity weight
WEIGHT_DATE = 0.1       # Date similarity weight
```

#### Thresholds

Modify similarity thresholds:

```python
# In build_event_clusters_from_faiss function
high_threshold = 0.75  # Adjust based on your data
```

### Data Format

#### Input Data (dataset.jsonl)

Each line should be a JSON object with:

```json
{
  "id": "unique_id",
  "text": "full text content",
  "summary": "summarized content", 
  "subject": "subject information",
  "location": "location information",
  "tags": ["tag1", "tag2"],
  "date": "YYYY-MM-DD",
  "source": "data source"
}
```

#### Ground Truth (new_incidents.jsonl)

Each line should be a JSON object with:

```json
{
  "incident_id": 1,
  "ids": [1, 2, 3, 4, 5]
}
```

#### Output Format

Each cluster is saved as a JSON object:

```json
{
  "incident_id": "cluster_0",
  "cases": ["id1", "id2", "id3"],
  "descriptions": ["desc1", "desc2", "desc3"],
  "metadata": [{"id": "id1", ...}, ...],
  "similarities": [0.95, 0.92, 0.88],
  "cluster_size": 3
}
```

### Key Components

#### TextVectorizer

Handles text-to-vector conversion using Xinference:

```python
vectorizer = TextVectorizer(xinference_url="url", model_name="bge-m3")
vectors = vectorizer.batch_get_embeddings(descriptions, batch_size=32)
```

#### VectorDatabase

Manages FAISS vector storage and similarity search:

```python
db = VectorDatabase(dim=768)
db.add_vectors(vectors, descriptions, metadata_list)
results = db.search(query_vector, k=200)
```

#### Similarity Functions

- `calculate_date_similarity()`: Date-based similarity
- `calculate_jaccard_similarity()`: Tag-based similarity
- `calculate_exact_match_similarity()`: Exact string matching
- `calculate_fuzzy_similarity()`: Fuzzy string matching using Levenshtein distance

### Performance Considerations

- **Batch Processing**: Use appropriate batch sizes for vector generation
- **Memory Management**: Large datasets may require chunked processing
- **Similarity Thresholds**: Adjust based on data characteristics
- **Index Type**: Currently uses FAISS Flat index for exact search

### Troubleshooting

#### Common Issues

1. **Xinference Connection**: Ensure Xinference service is running and accessible
2. **Memory Issues**: Reduce batch size for large datasets
3. **Path Issues**: Use forward slashes for cross-platform compatibility
4. **Encoding Issues**: Ensure UTF-8 encoding for text files


---

## 中文

本项目使用向量嵌入和相似性指标实现事件对齐和聚类。它提供了基于不同数据表示的多种事件聚类方法。

### 项目结构

```
event_alignment/
├── accuracy.py                    # 聚类结果评估指标
├── event_alignment_text.py        # 基于文本的事件对齐
├── event_alignment_summary.py     # 基于摘要的事件对齐  
├── event_alignment_metadata.py    # 基于元数据的事件对齐
├── data/                          # 数据文件
│   ├── dataset.jsonl             # 主数据集
│   ├── dataset_summary.jsonl     # 摘要数据集
│   └── new_incidents.jsonl       # 真实聚类标签
├── event_alignment_text/          # 文本对齐输出目录
├── event_alignment_summary/       # 摘要对齐输出目录
└── event_alignment_metadata/      # 元数据对齐输出目录
```

### 功能特性

- **多种对齐方法**：三种不同的事件聚类方法
  - 基于文本：使用完整文本内容进行向量化
  - 基于摘要：使用摘要内容进行向量化  
  - 基于元数据：使用元数据字段进行向量化

- **向量数据库**：基于FAISS的向量存储和相似性搜索

- **综合相似性指标**： 
  - FAISS向量相似性
  - 主体相似性（精确/模糊匹配）
  - 位置相似性
  - 标签相似性（Jaccard）
  - 日期相似性

- **评估指标**： 
  - 聚类准确率
  - 宏平均准确率
  - 归一化互信息（NMI）

### 环境要求

```bash
pip install faiss-cpu numpy tqdm xinference jsonlines scikit-learn scipy
```

### 使用方法

#### 1. 基于文本的事件对齐

```bash
python event_alignment_text.py
```

此方法：
- 使用完整文本内容进行向量化
- 权重配置：FAISS=1.0，其他=0.0
- 高相似性阈值：0.78

#### 2. 基于摘要的事件对齐

```bash
python event_alignment_summary.py
```

此方法：
- 使用摘要内容进行向量化
- 权重配置：FAISS=0.6，其他各=0.1
- 高相似性阈值：0.75

#### 3. 基于元数据的事件对齐

```bash
python event_alignment_metadata.py
```

此方法：
- 使用元数据字段进行向量化
- 权重配置：FAISS=0.6，其他各=0.1
- 高相似性阈值：0.6

#### 4. 评估结果

```bash
python accuracy.py
```

这将评估聚类结果与真实标签的对比，输出：
- 聚类准确率
- 宏平均准确率  
- 归一化互信息（NMI）

### 配置

#### 向量化器设置

在每个对齐文件中更新Xinference URL和模型名称：

```python
vectorizer = TextVectorizer(
    xinference_url="your_xinference_url",
    model_name="bge-m3"  # 或其他嵌入模型
)
```

#### 相似性权重

在每个对齐文件中调整权重：

```python
WEIGHT_FAISS = 0.6      # FAISS向量相似性权重
WEIGHT_SUBJECT = 0.1    # 主体相似性权重
WEIGHT_LOCATION = 0.1   # 位置相似性权重
WEIGHT_TAGS = 0.1       # 标签相似性权重
WEIGHT_DATE = 0.1       # 日期相似性权重
```

#### 阈值

修改相似性阈值：

```python
# 在build_event_clusters_from_faiss函数中
high_threshold = 0.75  # 根据数据特征调整
```

### 数据格式

#### 输入数据 (dataset.jsonl)

每行应该是一个JSON对象：

```json
{
  "id": "unique_id",
  "text": "full text content",
  "summary": "summarized content", 
  "subject": "subject information",
  "location": "location information",
  "tags": ["tag1", "tag2"],
  "date": "YYYY-MM-DD",
  "source": "data source"
}
```

#### 真实标签 (new_incidents.jsonl)

每行应该是一个JSON对象：

```json
{
  "incident_id": 1,
  "ids": [1, 2, 3, 4, 5]
}
```

#### 输出格式

每个聚类保存为一个JSON对象：

```json
{
  "incident_id": "cluster_0",
  "cases": ["id1", "id2", "id3"],
  "descriptions": ["desc1", "desc2", "desc3"],
  "metadata": [{"id": "id1", ...}, ...],
  "similarities": [0.95, 0.92, 0.88],
  "cluster_size": 3
}
```

### 核心组件

#### TextVectorizer

使用Xinference处理文本到向量的转换：

```python
vectorizer = TextVectorizer(xinference_url="url", model_name="bge-m3")
vectors = vectorizer.batch_get_embeddings(descriptions, batch_size=32)
```

#### VectorDatabase

管理FAISS向量存储和相似性搜索：

```python
db = VectorDatabase(dim=768)
db.add_vectors(vectors, descriptions, metadata_list)
results = db.search(query_vector, k=200)
```

#### 相似性函数

- `calculate_date_similarity()`: 基于日期的相似性
- `calculate_jaccard_similarity()`: 基于标签的相似性
- `calculate_exact_match_similarity()`: 精确字符串匹配
- `calculate_fuzzy_similarity()`: 使用编辑距离的模糊字符串匹配

### 性能考虑

- **批处理**：使用适当的批处理大小进行向量生成
- **内存管理**：大数据集可能需要分块处理
- **相似性阈值**：根据数据特征调整
- **索引类型**：目前使用FAISS Flat索引进行精确搜索



