# 🤖 LLM Comparison Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.42.1-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive comparison study of different Large Language Models (LLMs) for conversational AI, featuring real-time interactive chat capabilities with multiple models simultaneously.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🛠 Technologies Used](#-technologies-used)
- [📦 Installation](#-installation)
- [🚀 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [🔬 Models Compared](#-models-compared)
- [📊 Results](#-results)
- [🔮 Future Work](#-future-work)
- [🤝 Contributing](#-contributing)
- [👤 Contact](#-contact)

## 🎯 Project Overview

This project provides a systematic comparison of various pre-trained language models through an interactive chat interface. Users can simultaneously interact with multiple models to observe their different response patterns, capabilities, and specializations in real-time conversations.

## ✨ Features

- 🔄 **Multi-Model Comparison**: Compare 5 different LLMs side-by-side
- 💬 **Interactive Chat Interface**: Real-time conversation with all models
- 📊 **Response Analysis**: Observe diverse model behaviors and capabilities
- 🎯 **Specialized Models**: Include conversational, translation, and text-to-text models
- 🚀 **Easy Setup**: Simple installation and execution process
- 📝 **Comprehensive Documentation**: Well-documented code for research purposes

## 🛠 Technologies Used

- **Google Colab** - Cloud-based Jupyter environment with free GPU
- **Python 3.10+** - Programming language (pre-installed in Colab)
- **PyTorch 2.2.2** - Deep learning framework
- **Transformers 4.42.1** - Hugging Face model library
- **SentencePiece** - Text tokenization
- **TorchText 0.17.2** - Text processing utilities
- **NumPy 1.26** - Numerical computations (downgraded for compatibility)

## 📦 Installation & Setup

### Prerequisites
- Google Account (for Google Colab access)
- Stable internet connection
- Modern web browser

### Setup Instructions

#### Option 1: Google Colab (Recommended)
1. **Open in Colab:**
   - Click [here](https://colab.research.google.com/github/Net-AI-Git/LLMs-01---comparison/blob/main/model_comparison.ipynb) to open directly in Colab
   - Or upload the `model_comparison.ipynb` file to your Google Drive and open with Colab

2. **Runtime Setup:**
   ```python
   # The notebook automatically installs required packages:
   # !pip install transformers==4.42.1 sentencepiece torch==2.2.2 torchtext==0.17.2
   ```

3. **GPU Acceleration (Optional but Recommended):**
   - Go to Runtime → Change runtime type
   - Select "GPU" under Hardware accelerator

#### Option 2: Local Jupyter Environment
1. **Clone the repository:**
```bash
git clone https://github.com/Net-AI-Git/LLMs-01---comparison.git
cd LLMs-01---comparison
```

2. **Install dependencies:**
```bash
pip install torch==2.2.2 transformers==4.42.1 sentencepiece torchtext==0.17.2 jupyter
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook model_comparison.ipynb
```

## 🚀 Usage

### Running the Comparison

1. **Open the Google Colab Notebook** and run all cells in order
2. **Automatic Setup**: The notebook will automatically install all required packages
3. **Model Loading**: The system will download and load all models (this may take 5-10 minutes on first run)
4. **Start Interactive Chat**: Run the final cell to begin the chat interface
5. **Multi-Model Interaction**: Type your questions and see responses from all models simultaneously

### Example Interactions

**Simple Factual Query:**
```python
You: what is the largest city in the world by area?

blenderbot: I'm not sure, but it is the most populous metropolitan area in the United States.
nllb: በስፋት በዓለም ላይ ትልቁ ከተማ ምንድነው?
bart-base: what is the largest city in the world by area?
t5-base: san marino
t5-small: san francisco
```

**Complex Ethical Question:**
```python
You: if artificial intelligence surpasses human intelligence in all domains, who should be held morally accountable for its actions?

blenderbot: I'm not sure, but I do know that there is a lot of controversy surrounding it.
nllb: ሰው ሰራሽ የማሰብ ችሎታ በሁሉም ዘርፍ ከሰው ልጅ የማሰብ ችሎታ በላይ ከሆነ ለድርጊቱ በሥነ ምግባር ተጠያቂው ማን ነው?
bart-base: if artificial intelligence surpasses human intelligence in all domains, who should be held morally accountable for its actions?
t5-base: human
t5-small: a human

You: bye
Chatbots: Goodbye!
```

### Chat Commands
- Type your question and press Enter
- Use `quit`, `exit`, or `bye` to end the session
- All models respond simultaneously for easy comparison

## 📁 Project Structure

```
LLMs-01---comparison/
│
├── model_comparison.ipynb    # Main Colab notebook with comparison implementation
├── README.md                # Project documentation
├── results/                 # [TO ADD] Screenshots and analysis results
│   ├── chat_examples/       # [TO ADD] Example conversations
│   └── performance_metrics/ # [TO ADD] Model performance data
└── docs/                    # [TO ADD] Additional documentation
```

**Note:** This is a Google Colab project - all dependencies are installed directly within the notebook using `!pip install` commands.

## 🔬 Models Compared

| Model | Type | Size | Specialization |
|-------|------|------|----------------|
| **BlenderBot** | Conversational AI | 400M | Open-domain dialogue |
| **NLLB-200** | Translation | 600M | Multilingual translation |
| **BART-base** | Text Generation | Base | Text summarization & generation |
| **FLAN-T5-base** | Instruction Following | Base | Task-specific text generation |
| **FLAN-T5-small** | Instruction Following | Small | Lightweight task generation |

### Model Details

- **BlenderBot (facebook/blenderbot-400M-distill)**: Designed for engaging conversations
- **NLLB (facebook/nllb-200-distilled-600M)**: Specialized in cross-lingual translation
- **BART (facebook/bart-base)**: Effective for text understanding and generation
- **FLAN-T5 (google/flan-t5-base & small)**: Instruction-tuned for various NLP tasks

## 📊 Results

![image](https://github.com/user-attachments/assets/c6cf634e-91b8-4a02-8a85-eabd2b62ed46)


### Key Observations

**[TO ADD - After running experiments, document:]**
- Response quality patterns for different question types
- Model-specific strengths and weaknesses
- Performance metrics (response time, coherence, relevance)
- Behavioral differences in conversational scenarios

## 🔮 Future Work

- 📈 **Quantitative Evaluation**: Implement BLEU, ROUGE, and perplexity metrics
- 🎯 **Specialized Benchmarks**: Add domain-specific evaluation datasets
- 🔍 **Response Analysis**: Automated response quality assessment
- 🌐 **Web Interface**: Develop a web-based comparison tool
- 📊 **Visualization Dashboard**: Real-time performance metrics display
- 🧪 **A/B Testing Framework**: Systematic model comparison methodology

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Netanel Itzhak**
- 📧 Email: ntitz19@gmail.com
- 💼 LinkedIn: [linkedin.com/in/netanelitzhak](https://www.linkedin.com/in/netanelitzhak)
- 🐙 GitHub: [github.com/Net-AI-Git](https://github.com/Net-AI-Git)

## 🙏 Acknowledgments

- Hugging Face for providing the Transformers library and model hub
- Facebook AI Research for BlenderBot and NLLB models
- Google Research for FLAN-T5 models
- The open-source community for continuous improvements in NLP

---
⭐ **Star this repository if you found it helpful!**
