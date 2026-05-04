# Bi-Mamba: Towards Accurate 1-Bit State Space Models

Shengkun Tang, Liqun Ma, Haonan Li, Mingjie Sun, Zhiqiang Shen (2024)

## 🧩 Problem to Solve

본 논문은 최근 Transformer의 대안으로 주목받는 Selective State-Space Model (SSM)인 Mamba 아키텍처의 효율적인 배포 및 학습을 위한 극단적인 양자화 문제를 다룬다. Mamba는 시퀀스 길이에 대해 선형적인 계산 복잡도를 가지며 추론 시 KV 캐시로 인한 메모리 요구량이 적다는 강력한 장점이 있다. 그러나 모델의 크기가 계속해서 커짐에 따라 훈련 및 배포 단계에서 여전히 상당한 메모리 점유와 에너지 소비라는 과제가 남아 있다.

특히, Transformer 기반의 대규모 언어 모델(LLM)에서는 1비트 수준의 극단적인 양자화 연구가 활발히 진행되었으나, SSM 모델이나 Mamba 구조에 대해 저비트 양자화 또는 이진화(Binarization)를 적용했을 때 모델이 어떻게 동작하는지에 대한 연구는 거의 이루어지지 않았다. 따라서 본 연구의 목표는 Mamba 모델의 성능 저하를 최소화하면서 가중치를 1비트로 줄이는 Bi-Mamba 아키텍처를 설계하고, 이를 통해 메모리 사용량과 에너지 소비를 획기적으로 줄이는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 직관은 기존의 Post-Training Binarization (PTB) 방식이 Mamba 모델의 가중치 분포를 최적의 이진 가중치 분포에서 벗어나게 하여 심각한 성능 저하를 초래한다는 점이다. 이를 해결하기 위해 본 연구는 다음과 같은 설계를 제안한다.

첫째, Mamba-2 아키텍처에서 파라미터의 대부분(약 90% 이상)을 차지하는 선형 투영 행렬(Linear Projection Matrices)만을 선택적으로 이진화하는 전략을 사용한다. 임베딩 층과 같이 시맨틱 표현 능력이 중요한 부분은 고정밀도로 유지하여 압축률과 성능 사이의 균형을 맞춘다.

둘째, 학습 가능한 스케일 인자($\alpha$)와 시프트 인자($\beta$)를 도입한 FBI-Linear 모듈을 통해 이진 가중치의 표현력을 높인다.

셋째, 사후 양자화가 아닌, 사전 학습된 고정밀도 교사 모델로부터 지식을 전달받는 Autoregressive Distillation 기반의 Binarization-aware training을 수행하여 처음부터 1비트 모델을 학습시킨다.

## 📎 Related Works

기존의 양자화 연구는 크게 두 가지 방향으로 나뉜다. Post-Training Quantization (PTQ)은 학습이 완료된 모델에 양자화를 적용하는 방식으로, GPTQ와 같은 방법들이 대표적이다. 하지만 이러한 방식은 4비트 미만의 저비트 설정에서 모델 성능이 급격히 하락하는 경향이 있다. 또 다른 방식인 Quantization-Aware Training (QAT)은 학습 과정에서 양자화 오차를 고려하지만, LLM의 거대한 크기로 인해 처음부터 학습시키기에 계산 비용이 너무 크다는 단점이 있다.

최근의 SSM 연구인 Mamba와 Mamba-2는 Transformer의 이차 복잡도 문제를 해결하며 우수한 성능을 보였으나, 이들에 대한 저비트 양자화 연구는 매우 부족한 실정이다. 본 논문은 기존 Transformer 대상의 이진화 기법(Bi-LLM, PB-LLM 등)을 Mamba에 적용했을 때, 가중치 분포의 미스얼라인먼트(Misalignment)가 발생하여 성능이 심각하게 저하됨을 지적하며, 이를 극복하기 위한 학습 기반의 이진화 프레임워크를 제시한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 이진화 범위
본 논문은 Mamba-2를 기본 아키텍처로 채택한다. 분석 결과, Mamba-2 모델 파라미터의 대부분은 입력 투영(Input Projection)과 출력 투영(Output Projection) 행렬에 집중되어 있다. 따라서 Bi-Mamba는 이 두 선형 모듈만을 이진화하고, Embedding, Layer Norm, Conv-1d, $\Delta$ bias 등의 모듈은 고정밀도로 유지한다.

### FBI-Linear 모듈
기존의 선형 층을 FBI-Linear 모듈로 대체한다. 이 모듈은 $\{1, -1\}$ 값만 가지는 이진 행렬 $W^b \in \mathbb{R}^{m \times n}$와 학습 가능한 고정밀도 스케일 인자 $\alpha \in \mathbb{R}^n$, 시프트 인자 $\beta \in \mathbb{R}^n$로 구성된다. 추론 과정은 다음과 같이 정의된다.

$$y = f_{W^b} x$$

여기서 $f_{W^b}$의 각 열(column) $\cdot,i$에 대한 연산은 다음과 같이 수행된다.

$$f_{W^b}_{\cdot,i} = \alpha_i W^b_{\cdot,i} + \beta_i$$

이 구조를 통해 단순히 $\pm 1$만 사용하는 것이 아니라, 각 열에 대해 적절한 스케일과 오프셋을 부여함으로써 이진 가중치의 표현력을 확장한다.

### 학습 목표 및 절차
Bi-Mamba는 처음부터 학습되지만, 고정밀도 교사 모델(Teacher Model, LLaMA2-7B)의 출력을 모방하는 autoregressive distillation 손실 함수를 사용한다. 손실 함수는 학생 모델($p^S$)과 교사 모델($p^T$) 간의 교차 엔트로피(Cross-Entropy)로 정의된다.

$$L_{\text{Bi-Mamba}} = -\frac{1}{n} \sum_{k=1}^{n} p^T(x_{k+1}) \cdot \log p^S(x_{k+1})$$

이때 이진 가중치 $W^b$는 학습 가능한 고정밀도 행렬 $W^f$에 $\text{sign}(\cdot)$ 함수를 적용하여 생성된다. $\text{sign}(\cdot)$ 함수는 미분 불가능하므로, 역전파 단계에서는 Straight Through Estimator (STE)를 사용하여 기울기를 근사적으로 전달한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Amber 데이터셋(RefinedWeb, StarCoder, RedPajama-v1 포함, 총 1.26T 토큰)을 사용하여 학습하였다.
- **모델 규모**: 780M, 1.3B, 2.7B의 세 가지 크기로 구현하였다.
- **평가 지표**: zero-shot 정확도(BoolQ, PIQA, HellaSwag, Winogrande, ARC, OpenbookQA)와 Perplexity(Wiki2, PTB, C4)를 측정하였다.
- **비교 대상**: Full-precision Mamba-2, GPTQ (3-bit, 2-bit), Bi-LLM (PTB 방식)을 기준선으로 설정하였다.

### 주요 결과
1. **정량적 성능**: Bi-Mamba는 모든 모델 크기에서 GPTQ-2bit 및 Bi-LLM보다 월등히 높은 정확도와 낮은 Perplexity를 기록하였다. 특히 2.7B 모델의 경우, 평균 zero-shot 정확도 $49.3\%$를 달성하여 GPTQ-2bit($35.8\%$)와 Bi-LLM($38.5\%$)을 크게 앞질렀다.
2. **언어 모델링 능력**: Perplexity 측정 결과, Bi-Mamba는 교사 모델 및 Full-precision 모델에 근접한 수치를 보였으며, 특히 C4 데이터셋에서는 일부 구간에서 Full-precision 모델보다 더 낮은 Perplexity를 기록하기도 하였다.
3. **저장 효율성**: Full-binarization을 적용했을 때 저장 공간의 $80\%$ 이상을 절감하였다. 예를 들어, Mamba-2 2.7B의 저장 크기를 $5.03\text{GB}$에서 $0.55\text{GB}$로 줄여 약 $89\%$의 압축률을 달성하였다.

## 🧠 Insights & Discussion

### 가중치 분포 분석
본 논문은 PTB 방식과 Binarization-aware training의 차이를 가중치 분포 시각화를 통해 분석하였다. PTB 방식은 이진화 후 가중치 분포가 최적의 지점에서 크게 벗어나는 경향이 있으나, Bi-Mamba의 방식은 학습 과정을 통해 이진화된 가중치가 원래의 고정밀도 가중치 분포와 유사하게 유지되도록 강제한다. 이는 저비트 표현에서도 모델의 역량을 최대한 보존할 수 있는 핵심 이유이다.

### 생성 능력의 한계
정량적 지표는 우수하지만, 실제 텍스트 생성 사례(Generation Case)에서는 일부 한계가 관찰되었다. Full-precision Mamba-2와 마찬가지로 Bi-Mamba 역시 동일한 구절을 반복해서 생성하는 경향이 나타났다. 다만, GPTQ-2bit나 Bi-LLM이 의미 없는 기호나 깨진 텍스트를 생성하는 것과 달리, Bi-Mamba는 일관성 있는 의미를 유지하며 텍스트를 생성한다는 점에서 훨씬 견고함을 입증하였다.

### 비판적 해석
본 연구는 1비트 SSM의 가능성을 열었으나, 학습 과정에서 여전히 고정밀도 교사 모델에 의존하는 Distillation 방식을 사용한다. 또한, 실제 하드웨어 가속 없이 시뮬레이션된 1비트 연산을 사용했으므로, 실제 추론 속도 향상을 위해서는 전용 하드웨어 설계가 뒷받침되어야 한다는 전제가 필요하다.

## 📌 TL;DR

Bi-Mamba는 Mamba-2 아키텍처의 선형 투영 층을 1비트로 이진화하고, 이를 Autoregressive Distillation으로 학습시킨 최초의 1-bit SSM 프레임워크이다. 이 연구는 사후 양자화(PTQ) 방식의 한계를 극복하고, 메모리 사용량을 $80\%$ 이상 줄이면서도 Full-precision 모델에 근접한 성능을 유지할 수 있음을 보였다. 이는 향후 저전력·저메모리 환경을 위한 SSM 기반 LLM의 하드웨어 가속 및 효율적 배포 연구에 중요한 기초가 될 것으로 기대된다.