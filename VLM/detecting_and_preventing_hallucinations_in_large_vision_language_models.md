# Detecting and Preventing Hallucinations in Large Vision Language Models

Anisha Gunjal, Jihan Yin, Erhan Bas (2024)

## 🧩 Problem to Solve

본 논문은 Instruction tuning을 거친 Large Vision Language Models(LVLMs)가 시각적 근거가 부족한 텍스트를 생성하는 **Hallucination(환각)** 현상을 해결하고자 한다. 특히 최신 모델인 InstructBLIP조차 생성된 텍스트의 약 30%가 존재하지 않는 객체, 부정확한 묘사, 잘못된 관계 설정 등의 환각을 포함하고 있다는 점을 지적한다.

이러한 환각 현상은 LVLM의 신뢰성과 정확성을 저해하며, 특히 실제 환경에서 모델을 적용할 때 치명적인 문제가 될 수 있다. 그러나 멀티모달 환경에서의 환각은 프로그램적으로 탐지하기 어렵고 인간의 직접적인 감독이 필요하여 비용이 많이 든다는 한계가 있다. 따라서 본 연구의 목표는 환각을 정밀하게 탐지할 수 있는 데이터셋을 구축하고, 이를 통해 환각을 억제할 수 있는 최적화 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **M-HalDetect 데이터셋 구축**: 상세한 이미지 묘사 텍스트에 대해 sub-sentence(문장 하위 단위) 수준의 세밀한 어노테이션을 제공하는 최초의 종합적인 멀티모달 환각 탐지 데이터셋을 제안한다. 이는 단순히 객체의 존재 여부뿐만 아니라, 객체의 속성 묘사와 관계의 정확성까지 포함한다.
2. **Fine-grained Direct Preference Optimization (FDPO) 제안**: 기존 DPO가 텍스트 쌍(pair)의 상대적 선호도를 이용하는 것과 달리, M-HalDetect의 세밀한 어노테이션 정보를 직접 활용하여 모델을 최적화하는 새로운 방법론을 제시한다.
3. **멀티모달 Reward Model 및 Rejection Sampling 적용**: InstructBLIP 기반의 Reward Model을 학습시켜 Rejection Sampling(RS)을 통해 환각을 줄이는 방안을 제시하였으며, 이 Reward Model이 LLaVA, mPLUG-OWL과 같은 다른 LVLM의 환각을 줄이는 데에도 일반화될 수 있음을 입증하였다.

## 📎 Related Works

**Large Vision Language Models (LVLMs)**
최근 LVLM은 시각적 백본을 사전 학습된 LLM에 결합하고 instruction tuning을 통해 제로샷 성능을 높이는 방향으로 발전해 왔다. LLaVA나 InstructBLIP 등이 대표적이며, 이들은 다양한 멀티모달 태스크에서 뛰어난 성능을 보이지만 환각 문제는 여전히 해결해야 할 과제로 남아 있다.

**LVLM에서의 환각 분석**
기존 연구인 POPE는 생성된 텍스트에 대해 질문을 던지는 방식으로 객체 환각(object hallucination)을 측정하였으며, LRV 데이터셋은 강건성을 높이기 위한 긍정/부정 지시문을 도입하였다. 그러나 이러한 기존 방식들은 객체의 존재 여부에만 집중하며, 객체 간의 상대적 위치나 복잡한 추론 과정에서 발생하는 세밀한 환각은 충분히 다루지 못했다.

**인간 선호도 정렬 (Alignment to Human Preferences)**
RLHF(Reinforcement Learning from Human Feedback)와 DPO(Direct Preference Optimization)가 LLM의 성능 향상과 환각 방지에 사용되어 왔다. 하지만 기존의 DPO나 RLHF는 전체 문장 단위의 선호도를 다루기 때문에, 문장의 어느 부분이 정확하고 어느 부분이 잘못되었는지에 대한 세밀한 해석력(interpretability)이 부족하다는 한계가 있다.

## 🛠️ Methodology

### M-HalDetect 데이터셋

본 데이터셋은 COCO 데이터셋의 `val2014` split에서 추출한 4,000장의 이미지와 이에 대해 InstructBLIP가 생성한 4개의 응답(총 16,000개 쌍)으로 구성된다. 각 응답은 sub-sentence 수준에서 다음 네 가지 카테고리로 분류된다.

- **Accurate**: 객체가 실제로 존재하며 묘사와 관계가 정확한 경우.
- **Inaccurate**: 객체가 존재하지 않거나 묘사가 틀린 경우, 혹은 분석 내용이 타당하지 않은 경우.
- **Analysis**: 시각적 근거보다는 주관적인 해석이나 복잡한 추론이 포함된 경우.
- **Unsure**: 판단이 모호한 경우.

### Multi-Modal Reward Model

환각을 탐지하기 위해 InstructBLIP의 가중치와 아키텍처를 재사용하고, 최종 embedding layer를 classification head로 교체한 Reward Model을 설계하였다.

- **학습 구조**: ViCuna를 디코더로 사용하며, QFormer를 통해 이미지 특징을 수용한다.
- **학습 방법**: 문장 수준(Sentence-level)과 세그먼트 수준(Segment-level)에서 학습을 진행하며, 각 세그먼트의 끝 토큰에 타겟 라벨을 부여하고 Cross-entropy loss를 통해 최적화한다.
- **분류 체계**:
  - **Binary**: Accurate와 Analysis를 하나로 묶어 $\text{Accurate}$ vs $\text{Inaccurate}$로 분류한다.
  - **Ternary**: $\text{Accurate}$, $\text{Inaccurate}$, $\text{Analysis}$ 세 가지로 분류한다.

### Rejection Sampling (RS)

학습된 Reward Model을 사용하여 여러 생성 결과 중 가장 환각이 적은 응답을 선택한다. 전체 패시지에 대해 각 문장의 non-hallucination negative log probabilities의 평균을 계산하여 점수를 매기며, $\text{best-of-}n$ ($n=16, 64$) 방식으로 최적의 응답을 추출한다.

### Fine-grained Direct Preference Optimization (FDPO)

전통적인 DPO는 선호하는 답변($y_w$)과 선호하지 않는 답변($y_l$)의 쌍을 필요로 하지만, FDPO는 단일 생성물 내의 세밀한 세그먼트 정보를 활용한다.

기존 DPO의 손실 함수는 다음과 같다:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta || \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} [\log \sigma(\Delta r)]$$
$$\Delta r = \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}$$

본 논문에서 제안하는 FDPO 손실 함수는 다음과 같이 정의된다:
$$\mathcal{L}_{\text{FDPO}}(\pi_\theta ; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y, c) \sim \mathcal{D}} [\log \sigma(\beta k)]$$
$$k = \begin{cases} -r & c=0 \\ r & c=1 \\ -\infty & c>1 \end{cases}, \quad r = \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)}$$
여기서 $x$는 현재 세그먼트 이전까지의 입력, $y$는 해당 세그먼트, $c$는 클래스이다. $c=1$은 선호되는 클래스(Accurate), $c=0$은 선호되지 않는 클래스(Inaccurate)를 의미하며, 그 외의 클래스(Analysis 등)는 학습에서 무시(ignore)한다.

## 📊 Results

### Reward Model 성능

Reward Model의 분류 성능(Accuracy, F1 Score)은 다음과 같이 나타났다.

- **Segment Level (Binary)**: Accuracy 83.92%, F1 83.22%
- **Segment Level (Ternary)**: Accuracy 77.2%, F1 76.93%
Binary 모델이 수치적으로는 더 높으나, Ternary 모델이 Accurate와 Analysis를 구분함으로써 Rejection Sampling 시 subjective analysis에 편향되는 현상을 방지하여 더 효과적임이 확인되었다.

### 환각 감소 효과 (Human Evaluation)

환각률 측정 방식은 $\frac{\text{Inaccurate words}}{\text{Total words (excluding analysis)}}$로 정의하였다.

- **InstructBLIP**:
  - **FDPO (Ignore Analysis)**: 환각률을 약 41% 감소시켰다.
  - **Rejection Sampling (Best-of-64)**: 환각률을 약 55% 감소시켰다.
- **타 모델로의 일반화 (Best-of-16 RS 적용)**:
  - **LLaVA**: 환각률 15% 감소.
  - **mPLUG-OWL**: 환각률 57% 감소.

### 분석 및 상관관계

Reward Model의 점수와 인간이 평가한 정확도 점수 사이에 강한 양의 상관관계가 있음이 확인되었다. 이는 제안된 Reward Model이 인간의 평가를 어느 정도 대체할 수 있는 효과적인 평가 지표가 될 수 있음을 시사한다.

## 🧠 Insights & Discussion

**강점 및 효과**
본 연구는 단순히 객체 존재 여부를 넘어 묘사와 관계라는 세밀한 관점에서 환각을 정의하고 이를 해결하기 위한 데이터셋과 방법론을 제시하였다. 특히 FDPO를 통해 추가적인 선호도 쌍(preference pairs) 구축 없이도 기존 데이터셋의 라벨만으로 모델을 최적화할 수 있음을 보였다. 또한, 특정 모델로 학습한 Reward Model이 다른 LVLM의 응답을 필터링하는 데 유효하다는 점은 매우 고무적이다.

**한계 및 비판적 해석**

1. **추론 속도**: Rejection Sampling은 환각 감소 효과가 가장 강력하지만, 추론 시 $n$배의 비용이 발생하므로 실시간 서비스 적용에는 한계가 있다.
2. **데이터 다양성 및 오버피팅**: 연구진이 언급했듯이, 단 한 번의 피드백 루프만 수행했기 때문에 모델이 학습 데이터의 목적에 오버피팅되어 이미지 묘사가 이전보다 일반적(generic)으로 변하거나 정밀도가 떨어지는 현상이 관찰되었다.
3. **분석 클래스의 처리**: FDPO 학습 시 Analysis 클래스를 부정적인 신호로 처리했을 때 오히려 성능이 저하되는 경향이 있었는데, 이는 모델이 억지로 문장을 늘리려다 새로운 환각을 생성하기 때문으로 분석된다. 이는 LVLM의 생성 메커니즘과 환각 사이의 복잡한 관계를 보여준다.

## 📌 TL;DR

본 논문은 LVLM의 세밀한 환각(객체 묘사 및 관계 오류)을 탐지하고 방지하기 위한 **M-HalDetect 데이터셋**과 **FDPO(Fine-grained DPO)** 방법론을 제안한다. 제안된 Reward Model과 RS 방식을 통해 InstructBLIP의 환각률을 최대 55%까지 낮췄으며, 이 성과가 LLaVA, mPLUG-OWL 등 다른 모델에도 전이됨을 입증하였다. 이 연구는 향후 LVLM의 신뢰성을 높이기 위한 정밀한 정렬(Alignment) 연구의 기초가 될 가능성이 높다.
