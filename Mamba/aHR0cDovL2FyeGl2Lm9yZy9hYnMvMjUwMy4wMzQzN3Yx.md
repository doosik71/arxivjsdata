# JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba

Xiaoyong Lu, Songlin Du (2025)

## 🧩 Problem to Solve

본 논문은 두 이미지 간의 정확한 대응점을 찾는 local feature matching 문제에서 성능과 효율성 사이의 균형을 맞추는 것을 목표로 한다. 기존의 최첨단 feature matcher들은 Transformer 기반의 attention 메커니즘을 사용하여 long-range dependency를 캡처하지만, 이는 공간 복잡도가 매우 높아 고해상도 이미지 처리 시 높은 추론 지연 시간(latency)과 방대한 학습 자원을 요구한다는 치명적인 단점이 있다.

특히 semi-dense 및 dense matching 방식은 격자점이나 모든 픽셀 간의 대응 관계를 구축하여 텍스처가 부족한 환경에서도 강건함(robustness)을 보이지만, Transformer의 연산 비용으로 인해 실시간 응용 프로그램에 적용하기 어렵다. 따라서 본 연구는 Mamba의 선형 복잡도 $O(N)$를 활용하여, 성능 저하를 최소화하면서도 모델 크기와 연산량을 획기적으로 줄인 ultra-lightweight matcher를 개발하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba를 feature matching 작업에 최적화하기 위해 제안된 **JEGO (Joint, Efficient, Global, Omnidirectional)** 스캔-병합(scan-merge) 전략과 이를 적용한 **JamMa** 모델이다. 

핵심 직관은 Mamba가 본래 단일 시퀀스 모델로 설계되어 이미지 쌍 간의 상호작용이 부족하고, 1D 전방 스캔만으로는 2D 이미지의 다방향성(omnidirectionality)과 전역 수용장(global receptive field)을 확보하기 어렵다는 점을 해결하는 것이다. 이를 위해 두 이미지를 교차로 스캔하는 Joint scan과 4방향 스킵 스캔, 그리고 국소 정보를 통합하는 Aggregator를 도입하여 Transformer 수준의 전역 문맥 파악 능력을 유지하면서 연산 효율성을 극대화하였다.

## 📎 Related Works

### Local Feature Matching
기존 연구는 크게 세 가지로 분류된다. 
- **Sparse matching**: SIFT, SuperPoint와 같은 검출기(detector)와 SuperGlue, LightGlue 같은 매칭 모델을 결합하여 사용한다. 하지만 검출기가 텍스처가 없는 영역에서 실패할 경우 전체 성능이 저하되는 한계가 있다.
- **Semi-dense matching**: LoFTR, ASpanFormer 등이 대표적이며, 격자(grid) 기반의 대응점을 먼저 찾고 이를 세밀하게 조정하는 coarse-to-fine 방식을 취한다. 텍스처 부족 환경에 강건하지만, attention 메커니즘의 높은 연산 비용이 문제이다.
- **Dense matching**: DKM, RoMa 등은 모든 픽셀에 대해 대응점을 추정하여 정확도가 가장 높으나, 파라미터 수와 실행 시간이 매우 길다.

### State Space Models (SSM)
Mamba는 입력 시퀀스에 따라 파라미터가 변하는 S6 모델을 통해 Transformer와 대등한 성능을 내면서도 선형 복잡도를 달성하였다. 이를 시각 분야에 적용한 Vim, VMamba, EVMamba 등이 제안되었으나, 이들은 주로 단일 이미지 작업(분류 등)에 집중되어 있으며, 두 이미지 간의 상호작용이 필수적인 feature matching 작업에 최적화된 설계는 부족한 상태였다.

## 🛠️ Methodology

### 전체 파이프라인
JamMa의 구조는 크게 세 단계로 구성된다: **CNN Encoder $\rightarrow$ Joint Mamba (JEGO 전략) $\rightarrow$ Coarse-to-Fine (C2F) Matching**.

### 1. Local Feature Extraction
ConvNeXt V2를 인코더로 사용하여 이미지 $I_A, I_B$로부터 coarse feature $F_c \in \mathbb{R}^{H_c \times W_c \times C_1}$와 fine feature $F_f \in \mathbb{R}^{H_f \times W_f \times C_2}$를 추출한다. 매우 가벼운 설정(0.65M 파라미터)에서도 경쟁력 있는 성능을 보였다.

### 2. Joint Mamba와 JEGO 전략
본 논문의 핵심인 Joint Mamba는 coarse feature의 상호작용을 위해 다음의 JEGO 전략을 사용한다.

- **JEGO Scan**: 
    - **Joint Scan**: 두 이미지의 특징 맵을 가로($X_h$)와 세로($X_v$)로 연결한 뒤, 두 이미지의 픽셀을 번갈아 가며 스캔한다. 이는 기존의 순차적 스캔(Sequential scan)보다 이미지 간의 고주파 상호작용(mutual interaction)을 훨씬 더 효과적으로 촉진한다.
    - **Efficient & Omnidirectional Scan**: EVMamba의 skip scan(일정 간격으로 픽셀을 건너뛰며 스캔)을 도입하여 시퀀스 길이를 $N/4$로 줄이면서, 동시에 4가지 방향(우, 좌, 상, 하)으로 스캔을 수행하여 전방향성을 확보한다.
- **JEGO Merge & Aggregator**: 
    - 4방향으로 처리된 시퀀스를 다시 2D 맵으로 복원하고 합산하여 $\tilde{F}_c$를 생성한다.
    - 하지만 개별 특징점은 여전히 국소적인 수용장만을 가진다. 이를 해결하기 위해 **Gated Convolutional Unit (GCU)** 기반의 Aggregator를 사용하여 $3 \times 3$ 윈도우 내에서 정보를 통합함으로써, 모든 특징점이 전역적이고 전방향적인 정보를 갖도록 만든다.
    - 수식은 다음과 같다:
    $$\sigma = \text{GELU}(\text{Conv}_3(\tilde{F}_c))$$
    $$\hat{F}_c = \text{Conv}_3(\sigma \cdot \text{Conv}_3(\tilde{F}_c))$$

### 3. Coarse-to-Fine (C2F) Matching
XoFTR의 모듈을 채택하여 정밀도를 높인다.
- **Coarse Matching**: coarse feature $\hat{F}_c$ 간의 내적을 통해 유사도 행렬 $S^c$를 구하고, row/column Softmax를 통해 매칭 확률 $P_{A \to B}, P_{B \to A}$를 계산한다.
- **Fine Matching**: coarse match 주변의 $5 \times 5$ 윈도우를 crop 하여 MLP-Mixer를 통해 상호작용시키고, Dual-Softmax를 통해 1:1 정밀 매칭 $M_f$를 수행한다.
- **Sub-pixel Refinement**: Tanh 활성화 함수를 사용하여 $\pm 1$ 범위의 오프셋 $\delta$를 회귀(regression) 예측하여 픽셀 이하 단위의 정밀도를 달성한다.

### 4. Supervision (Loss Functions)
전체 손실 함수는 세 가지의 합으로 구성된다:
- **Coarse & Fine Loss**: Ground-truth 매칭 행렬과 예측 확률 간의 Focal Loss($FL$)를 사용한다.
  $$L_c = FL(P^{gt}_c, P_{A \to B}) + FL(P^{gt}_c, P_{B \to A})$$
- **Sub-pixel Loss**: 대칭 에피폴라 거리(symmetric epipolar distance) 함수를 사용하여 정밀도를 감독한다.
  $$L_s = \frac{1}{|M_f|} \sum_{(x,y)} \|x^T E y\|^2 \left( \frac{1}{\|E^T x\|^2} + \frac{1}{\|E y\|^2} \right)^{0.2}$$
  (여기서 $E$는 ground-truth essential matrix이다.)

## 📊 Results

### 상대 포즈 추정 (Relative Pose Estimation)
MegaDepth 데이터셋에서 AUC@$5^\circ, 10^\circ, 20^\circ$ 지표와 효율성(Params, FLOPs, Time)을 측정하였다.
- **결과**: JamMa는 파라미터 수와 FLOPs 면에서 기존 attention 기반 matcher의 50% 미만을 사용하면서도, semi-dense 및 sparse matcher 중 최상위권의 성능을 보였다. 
- **효율성-성능 균형**: 6개 지표에 대한 평균 순위(Avg. Rank)에서 3.5위를 기록하며, 성능과 효율성의 최적 균형을 달성했음을 입증하였다.

### 호모그래피 추정 (Homography Estimation)
HPatches 데이터셋에서 corner point의 reprojection error AUC를 측정하였다.
- **결과**: ASpanFormer와 대등한 수준의 성능을 보이면서도, 파라미터 수는 ASpanFormer의 36% 수준에 불과하여 매우 효율적이다.

### Ablation Study
- **Joint Scan의 효과**: Sequential scan으로 대체했을 때 AUC@$5^\circ$ 기준 약 2.3%p 성능이 하락하여, 이미지 간 고주파 상호작용의 중요성이 확인되었다.
- **Aggregator의 필수성**: Aggregator가 없을 경우 전역 수용장(ERF)이 사라지며 성능이 급격히 저하된다.
- **Mamba 기반 모델 비교**: VMamba보다 3배 빠르고 성능은 더 우수하며, Linear Attention 기반 방식보다 7.6배 빠르면서 더 높은 정확도를 보였다.

## 🧠 Insights & Discussion

### 강점 및 통찰
JamMa는 Mamba의 선형 복잡도를 활용하면서도, 시각적 작업에서 Mamba가 겪는 '단방향성'과 '국소적 수용장' 문제를 JEGO 전략과 Aggregator로 영리하게 해결하였다. 특히, attention 연산을 Mamba로 대체함으로써 추론 시간의 병목 지점을 attention에서 Softmax 기반의 similarity matrix 계산 단계로 옮겼을 만큼 획기적인 속도 향상을 이루었다.

### 한계 및 비판적 해석
- **일반화 능력**: ScanNet 데이터셋(실내 장면)에서의 zero-shot 실험 결과, 실외 장면만큼의 성능이 나오지 않았다. 저자들은 이를 매우 적은 파라미터 수(ultra-lightweight)로 인한 표현력의 한계로 분석하고 있다. 이는 효율성을 극단적으로 추구한 결과가 일반화 성능의 트레이드-오프(trade-off)를 야기했음을 시사한다.
- **저해상도 이점**: Mamba의 특성상 시퀀스가 짧을 때(저해상도 이미지) 정보 소실(perceptual attenuation)이 적어, 저해상도 환경에서 타 모델 대비 월등한 성능 향상을 보인다는 점이 흥미롭다. 이는 리소스가 극도로 제한된 임베디드 환경에서 매우 유용한 특성이 될 수 있다.

## 📌 TL;DR

JamMa는 Mamba의 선형 복잡도를 local feature matching에 도입하여, 기존 Transformer 기반 모델보다 훨씬 가볍고 빠른 **ultra-lightweight semi-dense matcher**이다. 두 이미지를 교차 스캔하는 **Joint Mamba**와 4방향 스킵 스캔 및 Aggregator를 포함한 **JEGO 전략**을 통해 전역적/전방향적 특징 표현을 효율적으로 구현하였다. 결과적으로 파라미터와 연산량을 획기적으로 줄이면서도 최상위권의 매칭 성능을 유지하여, 실시간 로보틱스나 모바일 SLAM 시스템에 적용될 가능성이 매우 높다.