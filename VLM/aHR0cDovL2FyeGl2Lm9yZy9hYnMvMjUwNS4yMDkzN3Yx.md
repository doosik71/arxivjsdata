# On VLMs for Diverse Tasks in Multimodal Meme Classification

Deepesh Gavit, Debajyoti Mazumder, Samiran Das, Jasabanta Patro (2025)

## 🧩 Problem to Solve

본 연구는 멀티모달 밈(Meme) 분류 작업에서 Vision-Language Models(VLMs)의 성능을 체계적으로 분석하고 개선하는 것을 목표로 한다. 밈은 이미지와 텍스트가 상호보완적이거나 간접적으로 연결되어 있으며, 이를 정확히 이해하기 위해서는 단순한 시각적 인식뿐만 아니라 문화적 배경과 맥락에 대한 깊은 이해가 필수적이다.

기존의 딥러닝 모델들은 특정 데이터셋에 과적합되어 일반화 능력이 떨어지며, 일반적인 LLMs는 시각적 정보를 처리할 수 없다는 한계가 있다. 최신 VLMs 역시 밈의 복잡한 맥락적 이해 능력이 부족하여 실무 배포 수준의 높은 효용성을 달성하지 못하고 있다. 따라서 본 논문은 VLMs를 활용해 밈의 다양한 작업(풍자, 공격성, 감성 분류 등)을 효과적으로 수행할 수 있는 최적의 전략을 탐색하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 VLMs의 능력을 최대화하기 위해 프롬프팅, 어댑터 튜닝, 그리고 지식 증류(Knowledge Distillation) 관점의 접근 방식을 제안한 것이다.

1. **VLM 벤치마킹**: 다양한 프롬프팅 전략(Zero-Shot, Few-Shot, Chain-of-Thought)을 통해 여러 VLMs의 밈 분류 성능을 체계적으로 평가하였다.
2. **LoRA Fine-tuning 평가**: 파라미터 효율적 미세 조정 방식인 LoRA(Low-Rank Adaptation)가 밈 분류 성능 향상에 실질적인 도움이 되는지 분석하였다.
3. **CoVExFiL 방법론 제안**: VLM이 생성한 상세한 밈 해석 텍스트를 사용하여 소규모 LLM을 학습시키는 **Combining VLM Explanation to Fine-tune LLMs (CoVExFiL)** 접근 방식을 제안하였다. 이는 VLM의 시각적 추론 능력을 텍스트 형태로 추출하여 LLM의 분류 성능을 높이는 전략이다.

## 📎 Related Works

기존의 밈 분석 연구들은 주로 이미지와 텍스트 간의 상호작용을 모델링하는 데 집중해 왔다.

- **데이터셋**: MET-meme(은유적 밈), FigMemes(비유적 언어), Multi-OFF(공격적 콘텐츠) 등 특정 목적의 말뭉치들이 제안되었다.
- **모델 구조**: M2Seq2Seq-MLD, MSDBert, DISARM 등 모달리티 간의 불일치(Incongruity)나 교차 모달리티 어텐션을 활용한 모델들이 개발되었다.
- **한계점**: 기존 딥러닝 모델들은 학습 데이터 외의 밈에 대해 일반화 능력이 부족하며, 내부 추론 과정이 불투명한 '블랙박스' 형태여서 해석 가능성(Interpretability)이 떨어진다는 치명적인 단점이 있다.

본 논문은 이러한 한계를 극복하기 위해 최신 VLM의 추론 능력을 활용하고, 이를 다시 LLM으로 전이시키는 파이프라인을 구축함으로써 해석 가능성과 일반화 성능을 동시에 확보하려 한다.

## 🛠️ Methodology

본 연구는 세 가지 단계의 실험적 접근 방식을 취한다.

### 1. 프롬프팅 전략 (Experiment 1)

VLMs에 다음과 같은 네 가지 프롬프팅 방식을 적용하여 성능을 측정한다.

- **Zero-Shot (ZS)**: 예시 없이 바로 분류를 요청한다.
- **Zero-Shot Chain-of-Thought (ZSC)**: 단계별 추론 과정을 생성하도록 요청하여 논리적 근거를 마련하게 한다.
- **Few-Shot (FS)**: 몇 가지 입출력 예시를 제공하여 문맥을 학습시킨다.
- **Few-Shot Chain-of-Thought (FSC)**: 예시 제공과 단계별 추론을 결합하여 모델의 추론 능력을 극대화한다.

### 2. LoRA Fine-tuning (Experiment 2)

VLM의 전체 파라미터를 업데이트하는 대신, 저차원 행렬을 추가하여 학습하는 LoRA 방식을 적용한다. LoRA의 핵심은 가중치 업데이트 $\Delta W$를 두 개의 작은 행렬 $A$와 $B$의 곱으로 분해하여 학습하는 것이다.
$$\Delta W = B \times A$$
여기서 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$이며, $r \ll \min(d, k)$인 low-rank 값을 가짐으로써 학습 파라미터 수를 획기적으로 줄인다. 본 연구에서는 Attention 및 MLP 모듈에 LoRA 어댑터를 적용하였다.

### 3. CoVExFiL 파이프라인 (Experiment 3)

본 논문이 제안하는 핵심 방법론으로, 다음과 같은 2단계 절차로 구성된다.

- **Step 1 (VLM Interpretation)**: VLMs(예: Qwen2-VL)에 CoT 프롬프팅을 적용하여 밈에 대한 상세한 텍스트 해석(시각적 분석 $\rightarrow$ 텍스트 분석 $\rightarrow$ 통합 해석)을 생성한다.
- **Step 2 (LLM Fine-tuning)**: 위에서 생성된 'VLM 기반 해석 텍스트'를 입력값으로 하여, 소규모 LLM(BERT, RoBERTa, XLNet)을 분류기로 학습시킨다.

이 방식은 VLM이 가진 시각적 이해도를 텍스트라는 매개체로 변환하여, 텍스트 분류에 강점이 있는 LLM이 이를 효율적으로 처리하게 만드는 구조이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Memotion(감성, 유머, 풍자, 공격성, 동기부여 분류), MAMI(여성혐오 탐지 및 유형 분류)
- **사용 모델**:
  - VLMs: LLaVA-1.6, Qwen2-VL, LLaMA-3.2-Vision, InstructBLIP
  - LLMs: BERT, RoBERTa, XLNet
- **평가 지표**: 클래스 불균형 문제를 해결하기 위해 **Weighted F1 score (AWF1)**를 사용하였다.

### 주요 결과

- **프롬프팅 결과**: FSC(Few-Shot CoT) 전략이 가장 우수했으며, 특히 Qwen2-VL 모델이 높은 성능을 보였다.
- **LoRA 결과**: LoRA 튜닝은 예상외로 큰 성능 향상을 가져오지 않았다. 이는 제한된 파라미터 업데이트만으로는 밈의 핵심인 '문화적 맥락'을 학습하기 어렵기 때문으로 분석된다.
- **CoVExFiL 결과**: 제안 방법론이 가장 압도적인 성능을 기록하였다. 특히 SOTA 모델과 비교했을 때 **Sentiment(SN) 분류에서 26.14%의 성능 향상**을 보였으며, Sarcasm(SR)과 Offensiveness(OF)에서도 유의미한 상승을 기록하였다.

## 🧠 Insights & Discussion

### 분석 및 강점

본 연구는 VLM을 직접 분류기로 쓰는 것보다, VLM을 '해석기'로 쓰고 LLM을 '분류기'로 쓰는 분리 구조(Decoupled architecture)가 훨씬 효과적임을 입증하였다. 이는 VLM이 생성한 구조화된 설명(Explanation)이 LLM에게 풍부한 문맥 정보를 제공하여 분류 정확도를 높였기 때문이다.

### 한계점 및 비판적 해석

- **복잡한 뉘앙스 파악의 어려움**: 여전히 Sarcasm(풍자)과 Offensiveness(공격성) 작업에서는 성능 향상 폭이 낮았다. 이는 텍스트와 이미지가 정반대의 의미를 가지는 '시각적 아이러니'나 아주 미묘한 문화적 혐오 표현을 VLM이 완전히 포착하지 못함을 시사한다.
- **데이터 의존성**: GPT-4를 이용해 생성한 Silver-standard 설명을 기준으로 평가하였으나, 이는 GPT-4의 편향성이 반영되었을 가능성이 있다.
- **계산 비용**: 모든 모델의 Full fine-tuning 대신 LoRA를 선택했으나, 이로 인해 모델의 적응력이 제한되었을 가능성이 명시되었다.

## 📌 TL;DR

본 논문은 VLM을 이용한 밈 분류 성능을 높이기 위해 **VLM이 생성한 상세 해석 텍스트로 LLM을 학습시키는 CoVExFiL 전략**을 제안하였다. 실험 결과, 단순 프롬프팅이나 LoRA 튜닝보다 VLM의 추론 결과를 LLM에 전이시키는 방식이 훨씬 효과적이었으며, 특히 감성 분석 작업에서 비약적인 성능 향상을 거두었다. 이 연구는 향후 멀티모달 모델의 해석 가능성을 높이고 복잡한 사회 문화적 맥락을 이해하는 밈 분석 시스템 구축에 중요한 방향성을 제시한다.
