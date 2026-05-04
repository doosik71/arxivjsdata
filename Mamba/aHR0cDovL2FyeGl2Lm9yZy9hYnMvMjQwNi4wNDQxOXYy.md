# TSCMamba: Mamba Meets Multi-View Learning for Time Series Classification

Md Atik Ahamed, Qiang Cheng (2024)

## 🧩 Problem to Solve

본 논문은 다변량 시계열 분류(Multivariate Time Series Classification, TSC) 작업에서 기존 모델들이 간과해 온 시계열 데이터의 핵심적 특성인 Shift Equivariance(이동 등변성)와 Inversion Invariance(반전 불변성)를 효과적으로 캡처하지 못한다는 문제를 해결하고자 한다. 

Shift Equivariance는 입력 신호가 시간축으로 이동하더라도 모델이 동일한 패턴을 인식할 수 있게 하여 실제 데이터에서 빈번하게 발생하는 시간적 정렬 불일치 문제에 대한 회복력을 제공한다. 또한, Inversion Invariance는 시계열을 정방향과 역방향으로 읽었을 때 모두 유용한 특징을 추출함으로써 데이터 증강 효과와 노이즈에 대한 강건성을 높일 수 있다.

기존의 CNN은 Local한 특징 추출에는 능하나 Long-range dependency(장기 의존성) 해결에 한계가 있고, Transformer 계열은 시퀀스 길이에 따라 연산 복잡도가 제곱으로 증가하는 효율성 문제가 존재한다. 따라서 본 연구의 목표는 연산 효율성을 유지하면서 spectral(주파수) 및 temporal(시간) 영역의 다각적 뷰(Multi-view)를 통합하고, Mamba의 State Space Model(SSM)을 활용하여 장기 의존성을 효율적으로 모델링하는 TSC 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 시계열의 다각적 표현을 학습하는 Multi-view Learning과 Mamba의 선형 복잡도를 결합하고, 여기에 Inversion Invariance를 구현하기 위한 새로운 스캔 기법을 도입한 것이다.

1. **Multi-view Feature Extraction**: Shift Equivariance를 보장하는 Continuous Wavelet Transform(CWT) 기반의 주파수 특징과, ROCKET 및 MLP를 이용한 지역적(Local) 및 전역적(Global) 시간 특징을 동시에 추출하여 상호 보완적인 컨텍스트를 제공한다.
2. **Tango Scanning**: 단일 Mamba 블록 내에서 정방향 시퀀스와 역방향 시퀀스를 모두 처리하고 그 결과를 융합하는 'Tango Scanning' 기법을 제안하여, 추가적인 파라미터 증가 없이 시퀀스의 모든 쌍(Pairwise) 간 상호작용을 캡처하고 Inversion Invariance를 달성한다.
3. **Efficient Sequence Modeling**: Mamba의 Selective State Space mechanism을 도입하여 시계열 데이터의 긴 시퀀스를 선형 복잡도로 처리함으로써, 기존 Transformer 기반 모델 대비 연산 비용(FLOPs)을 획기적으로 줄이면서도 성능을 향상시켰다.

## 📎 Related Works

논문에서는 TSC 접근 방식을 네 가지 범주로 나누어 설명하며 기존 연구의 한계를 지적한다.

1. **전통적 방법 (Traditional Methods)**: DTW(Dynamic Time Warping)나 XGBoost 같은 방법론은 단순한 정렬이나 통계적 특징에는 강하나, 복잡한 시계열 구조를 학습하는 데 한계가 있다.
2. **딥러닝 접근 방식 (CNN, RNN)**: CNN은 본질적으로 Shift Equivariance를 가지지만 수용 영역(Receptive Field)이 제한적이며, RNN/LSTM은 장기 의존성 학습 시 기울기 소실 문제와 순차적 연산의 비효율성이 존재한다.
3. **Transformer 기반 방법**: Attention 메커니즘을 통해 강력한 성능을 보이지만, 시퀀스 길이 $L$에 대해 $O(L^2)$의 복잡도를 가져 매우 긴 시계열 데이터 처리 시 메모리와 연산 비용이 과다하게 발생한다.
4. **상태 공간 모델 (SSM)**: S4나 Mamba와 같은 모델은 선형 복잡도로 장기 의존성을 처리할 수 있는 유망한 대안이다. 하지만 기존 SSM 연구들은 TSC 작업에 특화된 Inversion Invariance나 주파수-시간 영역의 통합 뷰 학습을 충분히 다루지 않았다.

## 🛠️ Methodology

TSCMamba의 전체 파이프라인은 spectral 및 temporal 특징 추출, 특징 융합, Mamba 기반의 추론 엔진, 그리고 최종 분류 단계로 구성된다.

### 1. Spectral Representation (CWT)
입력 신호의 주파수 특성을 캡처하기 위해 Continuous Wavelet Transform(CWT)을 사용한다. 특히 진폭과 위상을 효과적으로 캡처하는 Morlet wavelet을 사용하며, 수식은 다음과 같다.
$$\psi(t) = (\pi^{-1/4})(1-\frac{t^2}{\sigma^2}) \exp(-\frac{t^2}{2\sigma^2}) \cos(2\pi ft)$$
CWT를 통해 생성된 2D 표현은 Conv2D 레이어를 통한 Patch Embedding 과정을 거쳐 저차원의 특징 벡터 $W \in \mathbb{R}^{B \times D \times X}$로 투영된다.

### 2. Temporal Feature Extraction
시간 영역에서는 두 가지 상호 보완적인 방식을 사용한다.
- **Local Features (ROCKET)**: 무작위 합성곱 커널을 사용하여 다양한 스케일의 지역적 특징을 추출한다. 이는 비지도 방식으로 수행되며, 각 채널당 $X$ 길이의 특징 벡터 $V^L$을 생성한다.
- **Global Features (MLP)**: 전체 수용 영역을 가지는 1층 MLP를 통해 시계열의 전역적 패턴을 캡처하여 $V^G$를 생성한다.

### 3. Fusing Multi-View Representations
추출된 Spectral 특징 $W$와 선택된 Temporal 특징 $V$(Local 또는 Global)를 융합한다. 이때 학습 가능한 파라미터 $\lambda$를 도입하여 두 도메인의 균형을 조절하며, 융합 결과 $V^W$는 다음과 같이 계산된다.
$$ \{V^W\}_{ijk} = \lambda V_{ijk} * (2-\lambda)W_{ijk} \quad \text{or} \quad \{V^W\}_{ijk} = \lambda V_{ijk} + (2-\lambda)W_{ijk} $$
최종적으로 $W$, $V^W$, $V$를 연결(Concatenate)하여 $U \in \mathbb{R}^{B \times D \times 3X}$ 텐서를 생성한다. 이때 $V$를 $V^L$로 할지 $V^G$로 할지는 학습 가능한 Binary Mask 형태의 Switch mechanism이 결정한다.

### 4. Inference with Tango Scanning
융합된 텐서 $U$는 두 개의 Mamba 블록을 통과한다. Mamba의 핵심인 SSM은 다음과 같은 연속 시간 시스템으로 정의된다.
$$\frac{dh(t)}{dt} = A h(t) + B u(t), \quad z(t) = C h(t)$$
여기서 $h(t)$는 숨겨진 상태(Latent State)이다. 본 논문은 여기서 **Tango Scanning**이라는 새로운 스캔 방식을 제안한다.
- **절차**: 입력 시퀀스 $v$와 이를 반전시킨 시퀀스 $v^{(r)} = \text{Reverse}(v)$를 동일한 Mamba 블록에 입력한다.
- **융합**: 정방향 출력 $a$와 역방향 출력 $a^{(r)}$를 원본 시퀀스와 함께 요소별 덧셈(Element-wise addition)으로 결합한다.
$$ s^{(o)} = v \oplus a \oplus v^{(r)} \oplus a^{(r)} $$
이 과정은 시간(Time) 차원과 채널(Channel) 차원 모두에서 수행되어, 시간적 상호작용과 채널 간 상관관계를 모두 캡처한다.

### 5. Output Class Representation
최종 융합된 텐서 $z$에 대해 Depth-wise Pooling(Average 또는 Max pooling)을 수행하여 채널 정보를 집약한 후, 2층의 MLP를 통해 최종 클래스 로짓(Logits)을 생성하고 Cross-Entropy 손실 함수를 통해 학습한다.

## 📊 Results

### 실험 설정
- **데이터셋**: UEA archive의 30개 데이터셋 (기존 벤치마크 10개 + 추가 20개).
- **비교 대상**: TSLANet, TimesNet, Flowformer, ROCKET, DTW, XGBoost 등 20개의 SOTA 모델.
- **지표**: 분류 정확도(Accuracy), 평균 순위(Rank), 연산 복잡도(FLOPs).

### 주요 결과
- **정확도 향상**: 벤치마크 데이터셋에서 기존 최상위 모델인 TSLANet 대비 평균 4.01%에서 7.93%까지 정확도가 향상되었다.
- **강건성**: 특히 패턴이 불규칙한 EC, HW 데이터셋에서 타 모델 대비 월등한 성능 향상을 보였으며, 전체적으로 평균 순위에서 가장 높은 성적을 기록했다.
- **연산 효율성**: FLOPs 측정 결과, TimesNet이나 Flowformer 같은 Transformer 기반 모델보다 압도적으로 낮은 연산 비용을 보였다. (예: EC 데이터셋에서 TimesNet 1.11T FLOPs $\rightarrow$ TSCMamba 1.69G FLOPs로 극적인 감소).

## 🧠 Insights & Discussion

### 강점 및 해석
- **Tango Scanning의 효과**: 이론적 분석과 Attention Map 시각화를 통해, Tango Scanning이 단일 Mamba 블록만으로도 모든 토큰 쌍 간의 상호작용(Full Pairwise Attention)을 캡처할 수 있음을 증명하였다. 이는 Causal Masking으로 인해 미래 토큰을 보지 못하는 일반 Mamba의 한계를 극복한 것이다.
- **Multi-view의 시너지**: CWT를 통한 Shift Equivariance 확보와 ROCKET의 Local-Global 특징 결합이 시계열의 복잡한 도메인 특성을 잘 반영하고 있음을 Ablation Study를 통해 확인하였다. 특히 ROCKET의 비학습적 커널 특징이 성능 향상에 크게 기여함을 보였다.

### 한계 및 논의
- **메모리 요구량**: 융합 단계에서 생성되는 $B \times D \times 3X$ 크기의 텐서가 $X$ 값에 따라 연산 및 메모리 요구량을 증가시킬 수 있다는 점이 한계로 언급되었다.
- **CWT 의존성**: 현재는 Morlet wavelet 기반의 CWT만을 사용하고 있으나, 다른 spectral 변환 기법을 도입한다면 추가적인 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

본 논문은 시계열 분류(TSC)를 위해 **CWT 기반 주파수 뷰와 ROCKET/MLP 기반 시간 뷰를 통합**하고, 이를 **Mamba SSM**으로 처리하는 **TSCMamba**를 제안한다. 특히 **Tango Scanning**이라는 새로운 기법을 통해 Inversion Invariance를 달성하고 연산 복잡도를 선형으로 유지하면서도 SOTA 모델들을 상회하는 정확도를 달성하였다. 이 연구는 고효율·고성능 시계열 분류 모델의 새로운 방향성을 제시하며, 특히 매우 긴 시퀀스를 다루는 실시간 의료 및 금융 데이터 분석에 적용될 가능성이 높다.