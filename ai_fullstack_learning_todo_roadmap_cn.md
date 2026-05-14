# AI Full Stack 学习与项目落地路线图（MacBook Pro M4 Max）

> 目标：
>
> 从零开始，完整掌握：
>
> - 本地 LLM 部署
> - LoRA 微调
> - OCR + RAG
> - AI Agent
> - 企业 AI 系统落地
>
> 最终完成：
>
> ```text
> 本地LLM + RAG + Agent + ERP业务整合
> ```
>
> 打造完整 AI 工程能力。

---

# 一、整体学习路线

建议顺序：

```text
项目1：本地LLM
    ↓
项目2：OCR + RAG
    ↓
项目3：AI Agent
    ↓
项目4：企业AI工程化
```

建议周期：

| 阶段 | 时间 |
|---|---|
| 第一阶段：LLM基础 | 2~3周 |
| 第二阶段：RAG | 2~3周 |
| 第三阶段：Agent | 2~4周 |
| 第四阶段：工程化 | 持续迭代 |

---

# 二、项目1：本地 LLM（核心基础）

目标：

```text
下载模型 → 微调 → 推理 → API服务
```

最终得到：

```text
完全离线的本地中文 ChatGPT
```

---

# 2.1 环境搭建

## Todo

- [ ] 安装 Python 3.11
- [ ] 安装 Miniconda
- [ ] 创建 AI 虚拟环境
- [ ] 安装 JupyterLab
- [ ] 安装 Git LFS
- [ ] 安装 Homebrew
- [ ] 安装 VSCode / PyCharm

---

# 2.2 Transformer 基础（必须掌握）

## 学习内容

- [ ] Tokenizer
- [ ] Attention
- [ ] Transformer Block
- [ ] Embedding
- [ ] Position Encoding
- [ ] KV Cache
- [ ] Temperature
- [ ] TopP / TopK
- [ ] Streaming Generation

## 推荐目标

能够解释：

```text
LLM 为什么能生成文字
```

---

# 2.3 跑通 HuggingFace 模型

## 推荐模型

### 入门推荐

- Qwen2.5-1.5B
- Qwen2.5-3B

### 后续升级

- Qwen2.5-7B
- Llama 3.2

---

## Todo

- [ ] 安装 transformers
- [ ] 下载模型
- [ ] 本地执行 generate()
- [ ] 理解 tokenizer.encode/decode
- [ ] 理解 prompt template
- [ ] 实现 streaming 输出
- [ ] 测试不同 system prompt

---

# 2.4 LoRA 微调（重点）

## 学习内容

- [ ] LoRA 原理
- [ ] QLoRA 原理
- [ ] PEFT
- [ ] Instruction Tuning
- [ ] SFT（监督微调）

---

## 安装组件

- [ ] peft
- [ ] datasets
- [ ] accelerate
- [ ] bitsandbytes（后续 Linux 使用）

---

## Todo

- [ ] 准备 instruction 数据集
- [ ] 跑通最小 LoRA 微调
- [ ] 保存 adapter
- [ ] 加载 adapter 推理
- [ ] merge LoRA 权重
- [ ] 测试不同 LoRA rank

---

# 2.5 第一个业务微调项目

## 项目：Amazon Listing AI

训练数据示例：

```json
{
  "instruction": "生成亚马逊标题",
  "input": "无线蓝牙耳机",
  "output": "Bluetooth 5.3 Wireless Earbuds..."
}
```

---

## Todo

- [ ] 准备 100~500 条训练数据
- [ ] 微调电商助手
- [ ] 测试标题生成
- [ ] 测试五点描述生成
- [ ] Prompt 优化
- [ ] 输出 JSON 结构化结果

---

# 2.6 MLX（Mac 核心）

目标：

```text
最大化发挥 Apple Silicon 性能
```

---

## Todo

- [ ] 安装 MLX
- [ ] 安装 MLX-LM
- [ ] 转换 HuggingFace 模型为 MLX 格式
- [ ] 使用 MLX 推理
- [ ] 学习 quantization
- [ ] 测试 GPU 占用
- [ ] 测试不同量化精度

---

# 2.7 构建 OpenAI API 服务

目标：

```text
拥有自己的本地 ChatGPT API
```

---

## 技术栈

- FastAPI
- WebSocket
- Streaming

---

## Todo

- [ ] 创建 FastAPI 服务
- [ ] 实现 /chat/completions
- [ ] 支持 streaming
- [ ] 支持 history
- [ ] 支持 system prompt
- [ ] 实现多轮对话
- [ ] 接入 React Chat UI

---

# 三、项目2：OCR + RAG（企业 AI 核心）

目标：

```text
让 AI 拥有企业知识库
```

最终得到：

```text
图片/PDF → OCR → 向量化 → AI问答
```

---

# 3.1 OCR 学习

## 推荐框架

### 强烈推荐

- PaddleOCR

### 可选

- Tesseract
- EasyOCR

---

## Todo

- [ ] 安装 PaddleOCR
- [ ] 图片文字识别
- [ ] PDF OCR
- [ ] 表格 OCR
- [ ] 中英混合 OCR
- [ ] 商品包装 OCR
- [ ] OCR 结果结构化

---

# 3.2 第一个 OCR 业务项目

## 项目：商品包装解析

输入：

```text
商品包装图
```

输出：

```json
{
  "brand": "...",
  "model": "...",
  "capacity": "...",
  "features": []
}
```

---

## Todo

- [ ] OCR 提取文字
- [ ] LLM 提取结构化信息
- [ ] 输出 JSON
- [ ] 品牌风险词检测
- [ ] 侵权风险分析

---

# 3.3 Embedding（非常重要）

## 学习内容

- [ ] Embedding 原理
- [ ] 语义相似度
- [ ] Cosine Similarity
- [ ] Semantic Search
- [ ] Rerank

---

## 推荐模型

- bge-m3
- gte-Qwen2

---

## Todo

- [ ] 文本向量化
- [ ] 相似文本搜索
- [ ] 测试 TopK 检索
- [ ] 测试中英文混合检索

---

# 3.4 向量数据库

## 推荐

### 本地单机

- FAISS

### 企业级

- Milvus

---

## Todo

- [ ] 创建向量库
- [ ] 插入向量
- [ ] TopK 搜索
- [ ] Metadata Filter
- [ ] Hybrid Search
- [ ] 向量持久化

---

# 3.5 RAG（重点）

## 学习内容

- [ ] Chunk
- [ ] Retrieval
- [ ] ReRank
- [ ] Context Window
- [ ] Hallucination
- [ ] Prompt 拼接

---

# 3.6 第一个 RAG 系统

## 项目：Amazon 运营知识库

支持：

- PDF
- Word
- 商品资料
- Amazon规则

实现：

```text
AI 自动回答问题
```

---

## Todo

- [ ] 文档切块
- [ ] 向量化
- [ ] 建立索引
- [ ] 检索
- [ ] Prompt 拼接
- [ ] LLM 回答
- [ ] 多轮上下文

---

# 3.7 RAG 框架

## 推荐

- LangChain
- LlamaIndex

---

## Todo

- [ ] 跑通最小 Demo
- [ ] 自定义 Retriever
- [ ] 自定义 Prompt
- [ ] 自定义 Memory
- [ ] 自定义 Rerank

---

# 四、项目3：AI Agent（未来重点）

目标：

```text
让 AI 真正“干活”
```

---

# 4.1 Function Calling

## 学习内容

- [ ] Tool Calling
- [ ] JSON Schema
- [ ] Tool Router
- [ ] Tool Executor
- [ ] Tool Retry

---

# 4.2 第一个 Agent

## 项目：ERP AI 助手

用户：

```text
“查询美国库存”
```

AI 自动：

- 调数据库
- 调 API
- 生成结果

---

## Todo

- [ ] AI 调数据库
- [ ] AI 调 HTTP API
- [ ] AI 调 OCR
- [ ] AI 调 RAG
- [ ] AI 调本地函数

---

# 4.3 Agent 框架

## 推荐

- LangGraph
- AutoGen
- CrewAI

---

## Todo

- [ ] 单 Agent
- [ ] 多 Agent
- [ ] Workflow
- [ ] 状态机
- [ ] Retry 机制
- [ ] Memory

---

# 4.4 MCP（未来方向）

目标：

```text
AI 工具标准协议
```

---

## Todo

- [ ] 理解 MCP
- [ ] 实现 MCP Client
- [ ] 实现 MCP Tool
- [ ] AI 调本地工具
- [ ] AI 调 ERP

---

# 4.5 最终项目：跨境电商 AI 工作台

---

# 模块1：AI运营助手

## 功能

- [ ] Listing生成
- [ ] SEO优化
- [ ] 五点描述生成
- [ ] 侵权检测
- [ ] 标题审核

---

# 模块2：AI OCR助手

## 功能

- [ ] 商品包装解析
- [ ] 自动结构化
- [ ] PDF解析
- [ ] 图片信息抽取

---

# 模块3：AI Agent

## 功能

- [ ] 自动查库存
- [ ] 自动同步ERP
- [ ] 自动生成报表
- [ ] 自动翻译
- [ ] 自动数据分析

---

# 模块4：AI知识库

## 功能

- [ ] Amazon规则问答
- [ ] 产品手册问答
- [ ] ERP知识库
- [ ] 客服知识库

---

# 五、第四阶段：AI 工程化（进阶）

目标：

```text
接近真实企业生产环境
```

---

# 5.1 推理优化

## Todo

- [ ] Quantization
- [ ] GGUF
- [ ] KV Cache
- [ ] Streaming
- [ ] Batching
- [ ] Speculative Decoding

---

# 5.2 部署

## Todo

- [ ] Docker
- [ ] Docker Compose
- [ ] K8S 基础
- [ ] GPU部署
- [ ] 模型热更新
- [ ] 并发控制
- [ ] API Gateway

---

# 5.3 Linux + NVIDIA（后续进阶）

后续有 Linux + NVIDIA GPU 后：

## 学习内容

- [ ] vLLM
- [ ] TensorRT-LLM
- [ ] Triton Inference Server
- [ ] CUDA
- [ ] FlashAttention
- [ ] Continuous Batching

---

# 六、你的最终 AI 技术地图

---

# 第一层：模型层

- Qwen
- Embedding
- OCR
- LoRA

---

# 第二层：AI基础设施

- MLX
- 向量数据库
- 推理服务
- RAG

---

# 第三层：Agent层

- Tool Calling
- Workflow
- MCP
- Memory

---

# 第四层：业务层

- Amazon
- ERP
- OCR
- 自动化
- Agent

---

# 七、学习建议（非常重要）

不要：

```text
把所有理论学完再开始做项目
```

而应该：

```text
每学一个知识点
立即做业务功能
```

---

# 示例

## 学 LoRA

立即：

```text
做电商标题生成
```

---

## 学 RAG

立即：

```text
做 Amazon 规则知识库
```

---

## 学 Agent

立即：

```text
做 ERP AI 助手
```

---

# 八、最终目标

你最终真正值钱的不是：

```text
“知