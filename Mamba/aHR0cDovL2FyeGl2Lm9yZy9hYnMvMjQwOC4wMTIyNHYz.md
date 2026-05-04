# Multi-head Spatial-Spectral Mamba for Hyperspectral Image Classification

Muhammad Ahmad, Muhammad Hassaan Farooq Butt, Muhammad Usama, Hamad Ahmed Altuwaijri, Manuel Mazzara, Salvatore Distefano (2024)

## 🧩 Problem to Solve

본 논문은 초분광 이미지 분류(Hyperspectral Image Classification, HSIC)에서 발생하는 고차원 데이터 처리의 효율성과 정확도 사이의 트레이드오프 문제를 해결하고자 한다. 초분광 이미지는 수많은 좁은 밴드를 통해 세밀한 분광 정보를 캡처하므로 정밀한 물질 식별이 가능하지만, 데이터의 고차원성으로 인해 분석에 어려움이 따른다.

기존의 Transformer 기반 모델들은 Self-attention 메커니즘을 통해 공간적-분광적 특징 간의 장거리 의존성(long-range dependencies)을 잘 포착하지만, 연산 복잡도가 입력 크기의 제곱에 비례하는 quadratic computational complexity 문제를 가지고 있다. 이는 고차원 HSI 데이터를 처리할 때 실용적인 적용을 제한하며, 대량의 레이블링된 데이터가 없을 경우 오버피팅(overfitting)에 취약하다는 단점이 있다.

최근 등장한 State Space Model(SSM) 기반의 Mamba 모델은 Transformer 수준의 장거리 의존성 포착 능력을 유지하면서도 선형 시간 복잡도(linear time complexity)를 달성하여 효율적인 대안으로 주목받고 있다. 그러나 기존의 Mamba 모델들은 HSI 데이터의 풍부한 분광 정보를 충분히 활용하지 못하며, 특히 분광 밴드 간의 순차적 특성과 맥락 정보를 보존하는 데 어려움이 있다. 따라서 본 논문의 목표는 Mamba의 효율성을 유지하면서도 Multi-head Self-Attention(MHSA)과 토큰 강화 메커니즘을 결합하여 공간적-분광적 특징을 극대화한 MHSSMamba 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 효율적인 상태 공간 모델링에 Multi-head Self-Attention과 적응형 게이팅(Adaptive Gating) 메커니즘을 통합하여, HSI의 공간적-분광적 정보를 독립적이면서도 상호 보완적으로 추출하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **독립적 토큰 추출**: 입력 HSI 패치에서 분광 토큰(Spectral Token)과 공간 토큰(Spatial Token)을 분리하여 추출함으로써, 모델이 두 종류의 정보를 독립적으로 활용하고 특징 표현력을 높일 수 있도록 설계하였다.
2. **맞춤형 MHSA 메커니즘**: 분광 밴드와 공간적 위치 사이의 복잡한 관계를 효율적으로 캡처하기 위해, Dense 레이어를 통한 투영과 리셰이핑 과정을 포함한 맞춤형 Multi-head Self-Attention을 도입하였다.
3. **이중 게이팅 기반 특징 강화**: 학습된 게이팅 신호를 통해 분광 및 공간 토큰을 각각 강화하는 이중 게이트(dual-gate) 메커니즘을 도입하여, 공간적-분광적 맥락에 따라 특징을 적응적으로 정제한다.
4. **SSM을 통한 시퀀스 모델링**: 학습된 전이(transition) 및 업데이트 과정을 통해 상태 표현을 유지하고 갱신하는 SSM을 도입하여, HSI 데이터의 순차적 의존성과 템포럴 패턴을 효과적으로 모델링하였다.

## 📎 Related Works

본 논문은 HSIC를 위한 다양한 딥러닝 접근 방식을 다룬다. 전통적인 딥러닝(TDL) 방법론(CNN 등)은 최근 Transformer 기반 모델들로 대체되는 추세이며, Transformer는 공간-분광 특징의 장거리 의존성을 포착하는 데 탁월한 성능을 보였다. 하지만 앞서 언급한 연산 복잡도와 데이터 요구량 문제가 한계로 지적된다.

최근에는 Mamba(SSM)를 시각적 작업에 적용하려는 시도가 늘고 있으며, 특히 HSI 분야에서는 고차원 데이터를 효율적으로 처리하기 위한 방안으로 연구되고 있다. 기존의 Mamba 기반 모델들은 다음과 같은 한계를 가진다.
- **분광 정보의 소실**: Mamba 구조가 HSI의 풍부한 분광 정보를 간과하여 분광 특징 구별 능력이 떨어진다.
- **균형 부족**: 공간적 특징과 분광적 특징 사이의 균형을 맞추는 데 어려움이 있으며, 이로 인해 중요한 정보가 누락되는 경우가 발생한다.
- **맥락 유지의 어려움**: 분광 밴드 전반에 걸친 순차적 특성과 맥락 정보를 보존하는 능력이 부족하다.

MHSSMamba는 이러한 한계를 극복하기 위해 MHSA와 토큰 강화 모듈을 Mamba 프레임워크 내에 통합하여 차별성을 둔다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 토큰 생성
입력 HSI 데이터의 형태는 $(H, W, C)$이며, 이를 겹치는 3D 패치로 나누어 처리한다. 각 패치는 공간 토큰 $S$와 분광 토큰 $F$로 나누어 생성된다.
- **공간 토큰 $S$**: 2D 공간 데이터를 평탄화하여 생성하며, $S = [s_1, s_2, \dots, s_C] \in \mathbb{R}^{B \times (HW) \times C}$ 형태를 가진다.
- **분광 토큰 $F$**: 1D 분광 데이터를 평탄화하여 생성하며, $F = [f_1, f_2, \dots, f_{HW}] \in \mathbb{R}^{B \times (HW) \times C}$ 형태를 가진다.

### 2. 토큰 강화 (Token Enhancement)
HSI 샘플의 중심 영역($c$)을 컨텍스트로 사용하여 토큰의 중요도를 동적으로 조정하는 게이팅 메커니즘을 적용한다.

$$e_S^{(l)} = S^{(l)} \odot \sigma(W_s c + b_s)$$
$$e_F^{(l)} = F^{(l)} \odot \sigma(W_f c + b_f)$$

여기서 $\odot$은 요소별 곱셈(element-wise multiplication)이며, $\sigma$는 시그모이드 함수이다. $W_s, W_f$는 컨텍스트 $c$를 토큰 공간으로 투영하는 학습 가능한 가중치 행렬이고, $b_s, b_f$는 편향(bias) 항이다.

### 3. Multi-head Self-Attention (MHSA)
강화된 토큰 $e_S^{(l)}$와 $e_F^{(l)}$를 입력으로 하여 분광-공간 간의 복잡한 의존성을 학습한다. 쿼리($Q$)는 공간 토큰에서, 키($K$)와 밸류($V$)는 분광 토큰에서 생성한다. 각 헤드 $i$에 대해 다음과 같이 계산된다.

$$Q_i = e_S^{(l)} W_{Q_i}, \quad K_i = e_F^{(l)} W_{K_i}, \quad V_i = e_F^{(l)} W_{V_i}$$

어텐션 스코어 $A_i$와 최종 출력 $O$는 다음과 같다.

$$A_i = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right)$$
$$O_i = A_i V_i, \quad O = \text{Concat}(O_1, O_2, \dots, O_h)$$

### 4. State Space Model (SSM) 및 분류
강화된 토큰 시퀀스 $O = (E_1, E_2, \dots, E_T)$를 입력으로 하여 상태 전이를 계산한다.

$$h_t = \text{ReLU}(W_{\text{transition}} h_{t-1} + W_{\text{update}} E_t)$$

여기서 $h_t$는 시점 $t$에서의 은닉 상태(hidden state)이며, 이전 상태와 현재 토큰의 정보를 결합하여 업데이트한다. 최종적으로 선형 분류기를 통해 클래스 확률 $y$를 생성한다.

$$y = \sigma(h_t W_{\text{classifier}})$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: WHU-Hi-LongKou, University of Pavia (UP), Salinas (SA), University of Houston (UH)의 4개 공개 데이터셋을 사용하였다.
- **학습 환경**: Adam 옵티마이저, 학습률 0.001, 50 epoch, 배치 크기 256, Softmax 손실 함수를 사용하였다.
- **하이퍼파라미터**: Mamba 블록 임베딩 차원 64, MHSA 헤드 수 4, SSM 상태 차원 128, 패치 크기 $4 \times 4$를 기본으로 설정하였다.
- **평가 지표**: Overall Accuracy (OA), Average Accuracy (AA), Kappa coefficient ($\kappa$)를 사용하였다.

### 2. 주요 결과 및 비교 분석
MHSSMamba는 대부분의 데이터셋에서 SOTA(State-of-the-art) 모델들보다 우수한 성능을 보였다.

- **University of Houston 데이터셋**: OA 96.92%를 달성하였으며, 기존 Mamba 기반 모델인 SS-Mamba보다 OA 기준 2.62% 향상된 결과를 보였다. 특히 Grass-Stressed 클래스에서 99.84%라는 매우 높은 정확도를 기록하였다.
- **Pavia University 데이터셋**: OA 96.41%를 달성하며 SS-Mamba보다 소폭 향상된 성능을 보였다.
- **정성적 분석**: 다양한 패치 크기($2 \times 2$부터 $10 \times 10$까지)를 테스트한 결과, 적절한 패치 크기가 노이즈 억제와 전역적 특징 캡처 사이의 균형을 맞추는 것이 중요함을 확인하였다.

### 3. 연산 복잡도 분석
모델의 복잡도는 각 구성 요소의 합으로 정의되며, 전체 복잡도는 다음과 같다.
$$O(B \times H \times W \times C \times \text{outChannels}^2 + B \times L \times \text{embedDim}^2 + B \times \text{numheads} \times L^2 \times \text{headdim} + B \times L \times \text{outChannels}^2 + T \times B \times \text{stateDim}^2)$$
최악의 경우(Worst Case), 어텐션 스코어 계산으로 인해 $O(B \times L^2 \times \text{embeddim}^2)$의 복잡도가 발생할 수 있으나, 전반적으로 효율적인 구조를 유지한다.

## 🧠 Insights & Discussion

**강점 및 유효성:**
본 연구는 Mamba의 효율적인 시퀀스 모델링 능력에 Transformer의 강력한 특징 추출 능력(MHSA)을 성공적으로 결합하였다. 특히 분광 토큰과 공간 토큰을 분리하여 처리하고, 중심 영역을 이용한 게이팅 메커니즘을 통해 HSI 데이터의 특수한 구조(공간-분광 결합)를 효과적으로 활용했다는 점이 성능 향상의 주요 원인으로 분석된다.

**한계 및 논의사항:**
1. **패치 크기의 영향**: 실험 결과 패치 크기에 따라 성능 변동이 발생하며, 이는 데이터셋의 특성과 노이즈 수준에 따라 최적의 패치 크기가 달라짐을 시사한다. 이에 대한 일반적인 가이드라인 제시가 부족하다.
2. **데이터 효율성**: Transformer 기반 모델의 오버피팅 문제를 언급하였으나, 제안 모델이 적은 양의 학습 데이터(예: 5% split) 환경에서 기존 모델 대비 얼마나 더 강건한지에 대한 심층적인 분석은 다소 부족하다.
3. **연산 비용**: 이론적인 복잡도는 낮으나, MHSA 모듈이 추가됨에 따라 순수 Mamba 모델보다는 연산량이 증가했을 가능성이 크다. 실제 추론 속도(Inference time)에 대한 정량적 비교가 추가되었다면 더 설득력이 있었을 것이다.

## 📌 TL;DR

본 논문은 초분광 이미지 분류를 위해 **Multi-head Self-Attention과 토큰 강화 모듈이 통합된 Mamba 구조(MHSSMamba)**를 제안한다. 이 모델은 분광-공간 토큰을 분리 추출하고 이중 게이팅 메커니즘으로 특징을 강화한 뒤, SSM을 통해 효율적으로 시퀀스를 모델링한다. 실험 결과, University of Houston 및 Pavia University 등 주요 벤치마크 데이터셋에서 기존 SOTA 및 SS-Mamba 모델을 능가하는 정확도를 달성하였다. 이 연구는 고차원 HSI 데이터 처리에서 효율성과 정확도를 동시에 확보할 수 있는 새로운 아키텍처 방향성을 제시하며, 향후 원격 탐사 및 정밀 분석 분야에 기여할 가능성이 크다.