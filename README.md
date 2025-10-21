# 計算神經科學課程教材

國立陽明交通大學 LSNS30045 / 114-1 學期

**授課教師:** 陳俊仲 (Chun-Chung Chen)
**系所:** 生物科技系
**上課時間:** 每週三 09:00-12:00
**地點:** 圖書資訊大樓839室

---

## 簡介

這個儲存庫收集了計算神經科學課程的講義、作業和程式碼實現。內容涵蓋從基礎Python到神經模型模擬的完整學習路徑。

> **注意:** 這是學生在課程中的作業寫法與筆記，開源分享在網路上供參考。

---

## 課程內容

| 講座 | 主題 | 涵蓋內容 |
|------|------|---------|
| 第1講 | Python基礎 | NumPy、Matplotlib、資料處理 |
| 第2講 | 神經編碼 | 編碼模型、最大似然估計 |
| 第3講 | 尖峰統計 | 泊松過程、點過程分析 |
| 第4講 | 尖峰觸發平均 | STA、線性濾波、LNP模型 |
| 第5講 | 神經元模擬 | LIF模型、事件驅動演算法 |
| 第6講 | 電生理學 | 膜特性、離子通道 |
| 第7講 | Hodgkin-Huxley | 動作電位、模型實現 |

---

## 檔案結構

```
lec01/  ~ lec07/
├── code##.ipynb          # 講座程式碼與筆記
├── lec##-*.pdf           # 講座幻燈片
└── hw##/                 # 作業
    ├── hw##.pdf          # 作業題目
    ├── hw##-data.npz     # 資料檔案
    └── 作業解答          # .ipynb 或 .py
```

---

## 使用方式

### 安裝環境

```bash
git clone https://github.com/thc1006/nycu-computational-neuroscience-LSNS30045.git
cd nycu-computational-neuroscience-LSNS30045

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install numpy matplotlib scipy jupyter ipython brian2
```

### 執行筆記本

```bash
jupyter lab

# 或
jupyter notebook
```

然後進入 lec01/ 資料夾，開啟 code01.ipynb 開始

---

## 需要的套件

- **NumPy** - 數值計算
- **Matplotlib** - 繪圖視覺化
- **SciPy** - 科學計算、檔案I/O
- **Jupyter** - 互動式筆記本
- **Brian2** (可選) - 第7講神經模擬

---

## 核心主題

**基本概念**
- 神經編碼與解碼
- 尖峰列車分析
- 感受野與反向相關
- 神經模型(LIF、Hodgkin-Huxley)

**技術方法**
- 線性非線性泊松(LNP)模型
- 尖峰觸發平均(STA)
- 最大似然估計
- 事件驅動模擬

---

## 資料集

- **真實神經資料:** 蒼蠅H1神經元視覺反應 (MATLAB格式, 5.2 MB)
- **合成資料:** 白噪聲、泊松尖峰、電流刺激
- **總大小:** ~76 MB (包含講義)

---

## 相關資源

- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/) - 免費線上教科書
- [Neuromatch Academy](https://neuromatch.io/) - 計算神經科學課程
- [Coursera 計算神經科學](https://www.coursera.org/learn/computational-neuroscience)
- [Computational Neuroscience 教科書 (Dayan & Abbott)](http://www.gatsby.ucl.ac.uk/teaching/courses/snp/snp-2017.html)

---

## 授權

Apache License 2.0 - 可自由用於教育、研究與商業用途

---

## 致謝

感謝陳俊仲教授的精心教學與課程設計。

課程融合了經典神經科學模型(Hodgkin & Huxley、Gerstner等)與現代計算方法，是理解計算神經科學的重要課程。

---

## 搜尋關鍵詞

computational-neuroscience, neural-encoding, hodgkin-huxley, leaky-integrate-and-fire, spike-triggered-average, python-neuroscience, jupyter-notebooks, neural-coding, neuroscience-education, brian2
