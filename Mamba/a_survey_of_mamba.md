# A Survey of Mamba

Haohao Qu, Liangbo Ning, Rui An, Wenqi Fan, Tyler Derr, Hui Liu, Xin Xu, and Qing Li (2024)

## 🧩 Problem to Solve

본 논문은 현대 딥러닝의 핵심인 Transformer 아키텍처가 가진 고유한 한계점인 **계산 복잡도 문제**를 해결하고자 하는 최근의 연구 흐름을 분석한다. Transformer의 Attention 메커니즘은 입력 시퀀스 길이에 대해 이차 복잡도($O(L^2)$)를 가지며, 이는 특히 매우 긴 시퀀스를 처리해야 하는 추론(Inference) 단계에서 막대한 시간과 메모리 비용을 발생시킨다.

이러한 문제는 문서 단위의 기계 번역이나 장문 요약과 같은 작업에서 Transformer의 실용성을 제한한다. 따라서 본 연구의 목표는 Transformer에 필적하는 모델링 능력을 유지하면서도 시퀀스 길이에 대해 선형 복잡도($O(L)$)의 확장성을 제공하는 새로운 아키텍처인 **Mamba**와 그 관련 연구들을 체계적으로 정리하고 분석하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 Mamba 아키텍처를 중심으로 한 최신 연구들을 세 가지 주요 관점에서 종합적으로 분석한 것이다.

1. **Mamba 아키텍처의 진화 분석**: Mamba-1의 Selective State Space Model(SSM)과 하드웨어 최적화 알고리즘부터, Mamba-2에서 제안된 Structured State-Space Duality(SSD) 이론까지의 발전 과정을 상세히 다룬다.
2. **데이터 적응 기술 분석**: 텍스트와 같은 순차적 데이터(Sequential Data)뿐만 아니라 이미지, 그래프, 포인트 클라우드와 같은 비순차적 데이터(Non-Sequential Data)에 Mamba를 적용하기 위한 다양한 스캔(Scanning) 기법과 적응 전략을 분석한다.
3. **광범위한 응용 분야 탐색**: NLP, 컴퓨터 비전, 음성 분석, 신약 개발, 추천 시스템, 로보틱스 등 Mamba가 실제로 적용되어 성능 향상을 이룬 사례들을 체계적으로 분류하여 제시한다.

## 📎 Related Works

논문은 Mamba의 기반이 되는 세 가지 주요 아키텍처의 특성과 한계를 다음과 같이 설명한다.

- **Recurrent Neural Networks (RNNs)**: 내부 메모리를 통해 순차 데이터를 처리하는 데 능숙하지만, 가중치의 반복 곱셈으로 인해 정보가 희석되는 기울기 소실 문제(Vanishing Gradient)가 발생하며, 순차적 처리 특성상 병렬 계산이 불가능하여 학습 속도가 매우 느리다.
- **Transformers**: Self-Attention 메커니즘을 통해 전역적 의존성을 포착하며 병렬 학습이 가능하지만, 앞서 언급한 대로 시퀀스 길이에 따른 이차 복잡도로 인해 추론 효율성이 극히 낮다.
- **State Space Models (SSMs)**: 선형 시스템의 특성을 이용하여 순환(Recurrence)과 합성곱(Convolution) 형태의 계산을 모두 지원함으로써 병렬 학습과 효율적인 추론을 동시에 달성한다. 그러나 기존의 SSM은 **시불변성(Time-Invariance)** 특성으로 인해 입력 내용에 따라 동적으로 정보를 선택하는 능력이 부족하여 컨텍스트 인식 모델링 능력이 떨어진다는 한계가 있다.

## 🛠️ Methodology

### 1. Mamba-1: Selective SSM 및 하드웨어 최적화

Mamba-1은 기존 SSM의 시불변성 문제를 해결하고 하드웨어 효율성을 극대화하기 위해 세 가지 핵심 기술을 도입한다.

**가. HiPPO 기반 메모리 초기화**
장기 기억 능력을 강화하기 위해 HiPPO(High-order Polynomial Projection Operator) 이론을 적용한다. 특히 scaled Legendre measure(LegS)를 사용하여 과거의 모든 데이터를 균등하게 고려하도록 상태 행렬 $A$를 초기화함으로써, 정보의 손실을 최소화하며 긴 시퀀스의 맥락을 유지한다.

**나. 선택 메커니즘 (Selection Mechanism)**
입력 데이터 $x$에 따라 모델 파라미터가 동적으로 변하도록 하여, 불필요한 정보는 필터링하고 중요한 정보만 유지하는 '콘텐츠 인식(Content-aware)' 능력을 부여한다.
구체적으로, 간격 $\Delta$와 행렬 $B, C$를 입력 $x$의 함수로 정의한다.
$$B \rightarrow S_B = W_B x, \quad C \rightarrow S_C = W_C x, \quad \Delta \rightarrow S_\Delta = \tau_\Delta(W_\Delta x)$$
이후 Zero-Order Hold (ZOH) 이산화 과정을 통해 다음과 같이 변환된다.
$$\bar{A} = \exp(S_\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot S_\Delta S_B$$
이를 통해 Mamba는 Transformer의 Attention과 유사하게 입력값에 기반한 유연한 모델링을 수행한다.

**다. 하드웨어 인식 계산 (Hardware-aware Computation)**
선택 메커니즘의 도입으로 합성곱 계산이 불가능해지자, Mamba는 GPU 메모리 계층 구조를 최적화하는 두 가지 알고리즘을 제안한다.

- **Parallel Associative Scan**: 연산의 결합 법칙을 이용하여 계산 복잡도를 $O(N^2 d)$에서 $O(N/t)$ 수준으로 낮춘 병렬 스캔 알고리즘이다.
- **Memory Recomputation**: 순전파(Forward pass) 시 중간 상태를 저장하지 않고 역전파(Backward pass) 시에 다시 계산함으로써 GPU 메모리 사용량을 획기적으로 줄인다.

### 2. Mamba-2: Structured State-Space Duality (SSD)

Mamba-2는 SSM과 Attention 사이의 이론적 연결 고리인 **SSD**를 제안한다.

- **이론적 핵심**: SSM과 Attention 모두 '반분리 행렬(Semi-separable matrix)' 변환으로 볼 수 있음을 증명하였다.
- **구조적 단순화**: $A$ 행렬을 단위 행렬의 스칼라 배($\alpha I$)로 단순화하여 파라미터 수를 대폭 줄였다.
- **계산 최적화**: 블록 분해 행렬 곱셈(Block-decomposition matrix multiplication) 알고리즘을 통해 Mamba-1보다 2~8배 빠른 학습 속도를 달성하였다.

### 3. 데이터 적응 및 스캔 모드 (Scanning Modes)

비순차적 데이터(이미지 등)를 처리하기 위해 데이터를 토큰화하고 시퀀스로 배열하는 다양한 스캔 방식이 제안되었다.

- **Flatten Scan**: 데이터를 1차원 시퀀스로 펼쳐서 스캔한다.
  - *Bidirectional Scan*: 정방향과 역방향을 동시에 스캔하여 공간 인식을 강화한다.
  - *Sweeping Scan*: 4가지 방향(Cross Scan)이나 다방면(Omni Scan)으로 훑으며 전역 특징을 추출한다.
- **Stereo Scan**: 다차원적 관점에서 스캔한다.
  - *Hierarchical Scan*: 로컬에서 글로벌로(Macro-Micro) 계층적 스캔을 수행한다.
  - *Spatiotemporal Scan*: 공간과 시간 축을 모두 고려하여 비디오 데이터를 처리한다.

## 📊 Results

본 논문은 서베이 논문이므로 단일 실험 결과보다는 Mamba 기반 모델들의 정량적 성과를 종합하여 제시한다.

- **컴퓨터 비전 (Vision Mamba)**:
  - DeiT 대비 **2.8배 빠른 속도**를 기록하였다.
  - 고해상도 이미지($1248 \times 1248$) 특징 추출 시 GPU 메모리 사용량을 **86.8% 절감**하였다.
- **음성 분석 (SPMamba, DPMamba)**:
  - Transformer 기반 베이스라인 대비 모델 성능을 **13% 향상**시켰으며, 계산 복잡도를 **566% 감소**시켰다.
- **텍스트 요약 (LOCOST)**:
  - 희소 어텐션(Sparse Attention) 모델과 대등한 성능을 보이면서도 학습 시 메모리 사용량은 최대 **50%**, 추론 시에는 **87%**까지 줄였다.
- **인간 동작 생성 (Motion Mamba)**:
  - 디퓨전 기반 방법론 대비 FID 지표를 **50% 개선**하였으며, 생성 속도는 **4배 더 빨랐다**.

## 🧠 Insights & Discussion

### 강점 및 기여

Mamba는 Transformer의 강력한 표현력과 RNN의 효율적인 추론 속도를 결합한 혁신적인 아키텍처이다. 특히 하드웨어 레벨의 최적화를 통해 이론적인 선형 복잡도를 실제 하드웨어 성능으로 구현해냈다는 점이 매우 높게 평가된다.

### 한계 및 미해결 과제

- **모델링 능력의 격차**: 특정 복잡한 패턴 포착 능력에서는 여전히 Transformer 기반 언어 모델에 비해 열세인 경우가 존재한다.
- **메모리 손실**: 매우 긴 시퀀스에서 정보가 점진적으로 소실되는 현상이 보고되고 있다.
- **신뢰성 부족**: 모델의 결정 과정에 대한 설명 가능성(Explainability), 공정성(Fairness), 개인정보 보호(Privacy) 및 적대적 공격에 대한 강건성(Robustness) 연구가 아직 초기 단계에 머물러 있다.

### 비판적 해석 및 향후 방향

Mamba-2의 SSD 프레임워크는 SSM을 Attention의 특수한 형태로 재정의함으로써, 기존 Transformer 생태계에서 개발된 수많은 최적화 기법(PEFT, LoRA, RAG 등)을 SSM으로 전이시킬 수 있는 이론적 토대를 마련하였다. 향후 연구는 단순히 속도를 높이는 것을 넘어, Transformer가 가진 'In-context learning' 능력과 '신뢰성'을 어떻게 확보할 것인가에 집중되어야 할 것이다.

## 📌 TL;DR

본 논문은 Transformer의 이차 복잡도 문제를 해결하기 위해 등장한 **Mamba** 아키텍처를 총망라한 서베이 보고서이다. Mamba는 **선택적 상태 공간 모델(Selective SSM)**과 **하드웨어 인식 알고리즘**을 통해 선형 복잡도로 긴 시퀀스를 효율적으로 처리하며, 최근에는 **SSD(State Space Duality)** 이론을 통해 Attention과의 통합 가능성을 제시하였다. 이 연구는 Mamba가 NLP를 넘어 비전, 음성, 의료, 로보틱스 등 다양한 도메인에서 기초 모델(Foundation Model)로서의 가능성이 매우 높음을 시사한다.
