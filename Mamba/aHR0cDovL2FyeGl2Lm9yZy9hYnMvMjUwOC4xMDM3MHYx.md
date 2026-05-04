# eMamba: Efficient Acceleration Framework for Mamba Models in Edge Computing

Jiyong Kim, Jaeho Lee, Jiahao Lin, Alish Kanani, Miao Sun, Umit Y. Ogras, Jaehyun Park (2025)

## 🧩 Problem to Solve

본 논문은 최근 시퀀스 데이터 처리에 있어 Transformer 모델보다 계산 효율성이 뛰어난 State Space Model (SSM) 기반의 Mamba 모델을 리소스가 제한된 엣지 컴퓨팅(Edge Computing) 환경에 효율적으로 배포하기 위한 하드웨어 가속 프레임워크의 부재 문제를 해결하고자 한다.

Transformer 기반 모델, 특히 Vision Transformers (ViTs)는 높은 정확도를 제공하지만, 추론 과정에서 Key-Value 계산으로 인한 이차 복잡도(Quadratic Complexity)를 가지며, 이는 막대한 계산 자원과 전력 소모를 야기한다. Mamba는 이를 선형 시간 복잡도(Linear Time Complexity)로 해결하여 매우 유망한 대안으로 떠올랐으나, 현재까지 제안된 Mamba 가속기들은 주로 거대 언어 모델(LLM)에 최적화되어 있어, 전력과 면적이 극도로 제한된 엣지 장치에 적용하기에는 부적합하다는 한계가 있다. 따라서 본 연구의 목표는 엣지 플랫폼에 최적화된 end-to-end 하드웨어 가속 프레임워크인 **eMamba**를 설계하여, 정확도 손실을 최소화하면서 지연 시간(Latency), 면적(Area), 전력 소모(Power)를 획기적으로 줄이는 것이다.

## ✨ Key Contributions

eMamba의 핵심 아이디어는 하드웨어 친화적인 알고리즘 근사(Approximation)와 아키텍처 최적화를 결합하여 계산 병목을 제거하는 것이다.

1.  **하드웨어 친화적 연산 근사**: 계산 비용이 높은 Layer Normalization을 Range Normalization으로 대체하고, SiLU 활성화 함수와 지수 함수(Exponential)를 Piecewise Linear Approximation(구간별 선형 근사)으로 대체하여 하드웨어 구현 복잡도를 낮추었다.
2.  **근사 인식 신경망 구조 탐색 (Approximation-aware NAS)**: 근사 연산 도입으로 발생할 수 있는 성능 저하를 막기 위해, 하이퍼파라미터를 최적화하는 NAS를 수행하여 리소스 비용과 정확도 사이의 최적의 균형점(Pareto front)을 찾았다.
3.  **하이브리드 양자화 전략**: 일반 레이어에는 8-bit 양자화를 적용하되, 수치적 불안정성과 오버플로가 발생하기 쉬운 SSM 레이어에는 Scale-aware 양자화 및 고정밀도 중간 상태(Hidden State) 표현 방식을 도입하여 정확도를 유지하였다.
4.  **실리콘 검증**: FPGA(AMD ZCU102) 및 ASIC(GlobalFoundries 22nm) 구현을 통해 제안하는 프레임워크의 실제 하드웨어 성능을 입증하였다.

## 📎 Related Works

기존의 AI 가속 연구들은 주로 CNN의 데이터 로컬리티 최적화나 Transformer의 행렬 곱셈 효율화에 집중해 왔다. 최근 Mamba 모델을 위한 가속기 연구로 MARCA와 LightMamba 등이 제안되었으나, 이들은 모두 LLM(Large Language Models)을 대상으로 설계되었다. LLM 가속기는 시퀀스 길이가 매우 길어짐에 따라 발생하는 element-wise 연산의 비중을 줄이는 데 집중하므로, 상대적으로 가벼운 모델을 사용하는 엣지 시나리오에서는 그 효율성이 떨어지며, 하드웨어 리소스 요구량이 여전히 높다.

또한, MambaQuant나 Quamba와 같은 양자화 연구들이 존재하지만, 이들 역시 주로 NLP 벤치마크에 국한되어 있다. eMamba는 엣지 환경의 비전 작업(Fashion-MNIST, CIFAR-10, MARS)뿐만 아니라 일반적인 언어 작업(WikiText2)까지 확장하여 검증함으로써, 기존 연구들보다 범용적이고 리소스 효율적인 엣지 전용 가속 솔루션을 제공한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Hardware-friendly Range Normalization
기존의 Layer Normalization은 제곱근(Square Root)과 표준편차 계산이 필요하여 하드웨어 구현 시 비용이 매우 크다. eMamba는 이를 다음과 같은 **Range Normalization**으로 대체한다.

$$\hat{y}_i = \gamma \cdot \frac{x_i - \mu}{range(x_i - \mu)} + \beta$$

여기서 $range(x) = \max(x) - \min(x)$이다. 이 방식은 복잡한 산술 연산 대신 비교기(Comparator)를 사용하여 최소/최대값을 찾으므로 계산 효율성이 극대화된다. 또한 $\gamma$와 $\beta$를 학습 가능한 파라미터로 설정하여 정확도 손실을 보완하였다.

### 2. Non-linear Function Approximation
하드웨어에서 구현하기 까다로운 비선형 함수들을 구간별 선형 근사(Piecewise Linear Approximation)로 처리한다.
- **SiLU**: $[-7, 7]$ 구간을 17개의 선형 세그먼트로 나누어 근사하며, $-7$ 미만은 0으로, $7$ 초과는 입력값 그대로 처리한다.
- **Exponential ($\exp$)**: $[-4, 1]$ 구간을 11개의 선형 세그먼트로 근사하며, $1$ 초과는 상수로, $-4$ 미만은 0으로 처리한다.
- **Softplus**: 하드웨어 구현이 매우 무거운 Softplus 함수를 단순한 **ReLU**로 대체하여 구현 복잡도를 낮추었다.

### 3. SSM Layer & Pipelining
SSM의 핵심 상태 방정식은 다음과 같다.

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$
$$y_t = C_t h_t + D x_t$$

eMamba는 이를 토큰 단위로 처리하는 **Layer-wise Pipelining** 구조로 설계하였다. 각 레이어는 파이프라인의 한 단계로 작동하며, Ready/Valid 핸드셰이크 프로토콜을 통해 데이터를 전달한다. 특히 SSM의 재귀적 특성으로 인해 발생하는 계산 지연을 줄이기 위해 연산을 Unroll하여 구현하였다.

### 4. Quantization Strategy
전체 모델에 대해 대칭 유니폼 양자화(Symmetric Uniform Quantization)를 적용하며, 스케일 팩터를 2의 거듭제곱으로 제한하여 곱셈 연산을 비트 시프트(Bit-shift)로 대체하였다.
- **일반 레이어**: 가중치, 활성화 함수 모두 $\text{INT8}$을 사용한다.
- **SSM 레이어 (Scale-aware)**: 재귀 연산 시 발생하는 비트 폭 증가(Bit-width explosion)를 막기 위해, 중간 상태 $h_t$는 $\text{INT24}$로 계산하고 저장 시에는 $\text{INT17}$로 재양자화(Re-quantization)하여 저장한다. 출력 $y_t$는 다시 $\text{INT8}$로 변환된다.

### 5. Approximation-aware NAS
모델의 표현력과 리소스 비용 사이의 트레이드-오프를 최적화하기 위해 다음 다섯 가지 하이퍼파라미터를 탐색한다:
- $D$ (모델 차원), $E$ (확장 계수), $P$ (패치 크기), $N$ (상태 차원), $M$ (Mamba 블록 수).
이 과정에서 Range Normalization과 ReLU를 학습 단계부터 적용하여 하드웨어 제약 조건 하에서의 최적 구조를 도출한다.

## 📊 Results

### 1. 정확도 및 모델 크기 비교
Fashion-MNIST, CIFAR-10, MARS 데이터셋에서 ViT 모델과 비교한 결과, eMamba는 유사하거나 더 높은 정확도를 유지하면서 모델 크기를 획기적으로 줄였다.
- **Fashion-MNIST**: ViT 대비 파라미터 수를 $19.9\times$ 감소시켰음에도 정확도는 오히려 상승(87.6% $\rightarrow$ 90.2%, FP32 기준)하였다.
- **MARS**: $\text{INT8}$ 양자화 상태에서 ViT보다 작은 모델 크기($1.63\times$ 감소)로 유사한 RMSE(8.83cm vs 9.05cm)를 달성하였다.

### 2. 범용성 검증 (WikiText2)
언어 모델링 작업에서도 시퀀스 길이가 512에서 8,192까지 증가함에 따라 Perplexity가 매우 안정적으로 유지되었다. 특히 RNN과 LSTM이 시퀀스 길이에 따라 성능이 급격히 저하되는 것과 달리, eMamba는 ViT나 Ref. Mamba와 유사한 안정성을 보여 엣지 환경의 다양한 작업에 적용 가능함을 입증하였다.

### 3. 하드웨어 성능 (FPGA 및 ASIC)
MARS 데이터셋을 기준으로 측정된 하드웨어 결과는 다음과 같다.

- **FPGA (ZCU102)**: 
    - **지연 시간(Latency)**: CNN 가속기 대비 $5.62\times$ 낮고, ViT 가속기 대비 $4.95\times$ 낮다.
    - **처리량(Throughput)**: CNN 대비 $9.95\times$, ViT 대비 $2.22\times$ 높다.
- **ASIC (GF 22nm)**: 
    - **면적**: ViT 가속기 대비 $4.77\times$ 작다.
    - **전력 및 에너지**: 전력 소모는 $9.84\times$ 낮으며, 에너지 소비는 $48.6\times$ 적다.

## 🧠 Insights & Discussion

### 1. Range Normalization의 병목 현상
실험 결과, Range Normalization 레이어의 병렬화 수준(Compute Units 수)이 전체 프레임 지연 시간에 결정적인 영향을 미친다는 것이 밝혀졌다. 유닛 수를 1개에서 10개로 늘렸을 때 지연 시간이 선형적으로 감소하였으나, 10개 이후부터는 다른 레이어(최종 프로젝션 레이어 등)가 새로운 병목이 되어 이득이 감소하였다. 이는 파이프라인 설계 시 첫 번째 단계의 지연 시간이 전체 처리량의 상한선을 결정한다는 점을 시사한다.

### 2. 양자화의 수치적 안정성
SSM 레이어에서 모든 변수를 $\text{INT8}$로 고정했을 때 심각한 정확도 저하가 발생하였다. 이는 SSM의 재귀적 누적 연산이 수치적 범위를 빠르게 벗어나기 때문이다. 본 논문에서 제안한 $\text{INT24} \rightarrow \text{INT17}$ 재양자화 전략은 정밀도 손실과 오버플로 방지 사이의 적절한 타협점을 찾아 하드웨어 효율성과 모델 정확도를 동시에 잡은 핵심적인 설계라 판단된다.

### 3. 비판적 해석
제안된 방식은 매우 효율적이지만, Piecewise Linear Approximation을 사용하므로 근사 세그먼트의 수를 결정하는 과정에서 데이터셋의 분포에 의존적이다. 만약 입력 데이터의 분포가 학습/프로파일링 단계와 크게 다를 경우, 근사 오차가 증가하여 성능이 하락할 가능성이 있다. 다만, 본 논문에서는 이를 NAS를 통해 어느 정도 완화하였음을 보여주었다.

## 📌 TL;DR

본 논문은 엣지 컴퓨팅 환경에 최적화된 Mamba 가속 프레임워크 **eMamba**를 제안한다. Range Normalization 도입, 비선형 함수의 구간별 선형 근사, $\text{INT8}$ 기반의 하이브리드 양자화, 그리고 NAS를 통한 구조 최적화를 통해, ViT 대비 **파라미터 수를 최대 $19.9\times$ 줄이면서도 유사한 정확도**를 달성하였다. 하드웨어 구현 결과, ViT 가속기 대비 **지연 시간은 약 5배 낮고, 에너지 소모는 48.6배 적은** 압도적인 효율성을 입증하였으며, 이는 향후 리소스 제한적인 엣지 AI 장치에서 SSM 기반 모델을 실용적으로 배포하는 데 중요한 기준이 될 것으로 보인다.