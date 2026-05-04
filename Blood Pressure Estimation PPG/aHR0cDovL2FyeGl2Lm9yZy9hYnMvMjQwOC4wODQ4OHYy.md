# PITN: Physics-Informed Temporal Networks for Cuffless Blood Pressure Estimation

Rui Wang, Mengshi Qi, Yingxia Shao, Anfu Zhou, Huadong Ma (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 비침습적(non-invasive) 센서를 이용한 커프리스 혈압(Cuffless Blood Pressure, BP) 추정 시 발생하는 **데이터 부족 문제**이다.

전통적인 혈압 측정 방식은 커프(Cuff)를 이용해 팔을 압박하는 방식을 사용하지만, 이는 사용자에게 불편함을 주어 연속적인 모니터링에 부적합하다. 이를 해결하기 위해 생체 임피던스(Bioimpedance), 광혈류 측정(PPG), 밀리미터파(mmWave) 등 다양한 웨어러블 센서를 이용한 추정 방식이 연구되고 있다. 그러나 이러한 딥러닝 기반 모델들은 각 피험자(Subject)별로 개인화된 모델을 학습시키기 위해 상당한 양의 실제 혈압 데이터(Ground Truth)를 요구한다. 특히, 의료 등급의 정밀한 혈압 측정값은 침습적이거나 번거로운 절차를 수반하므로, 현실적으로 다량의 학습 데이터를 확보하는 것이 매우 어렵다.

따라서 본 연구의 목표는 **매우 제한된 양의 데이터만으로도 정밀한 개인별 혈압 추정이 가능한 Physics-Informed Temporal Network (PITN)**를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 물리 법칙 기반의 제약 조건(Physics-informed constraints)과 시계열 데이터의 시간적 특성 추출 능력을 결합하고, 적대적 학습 및 대조 학습을 통해 데이터 부족 문제를 극복하는 것이다.

1. **PITN (Physics-Informed Temporal Network) 제안**: 기존 PINN(Physics-Informed Neural Networks)이 좌표값을 입력으로 사용하여 시계열 데이터의 시간적 의존성을 무시한다는 점을 개선하기 위해, 1차원 신호를 2차원으로 변환하여 특징을 추출하는 **Temporal Block**을 도입하였다.
2. **적대적 훈련(Adversarial Training)을 통한 데이터 증강**: PGD(Projected Gradient Descent) 알고리즘을 사용하여 생리적 특성을 유지하는 적대적 샘플을 생성함으로써, 학습 데이터가 부족한 상황에서도 모델의 강건성(Robustness)을 높이고 데이터 증강 효과를 얻었다.
3. **대조 학습(Contrastive Learning) 도입**: 혈압 값이 유사한 샘플들을 잠재 공간(Latent Space)에서 가깝게 배치하고, 서로 다른 값들을 멀리 떨어뜨리는 Soft Constraint를 추가하여 심혈관 동역학의 변별력 있는 변화를 효과적으로 학습하게 하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **데이터 주도 방식(Data-driven methods)**: BiLSTM 기반의 HNN, Autoencoder, Transformer 기반 모델 등이 연구되었다. 이들은 데이터가 충분할 때는 효과적이지만, 생물학적 시스템의 도메인 지식을 활용하지 않으므로 개인화된 모델링 시 데이터 부족 문제에 매우 취약하다.
- **PINN (Physics-Informed Neural Networks)**: 물리 법칙(PDE 등)을 손실 함수에 반영하여 적은 데이터로도 학습이 가능함을 보였다. 하지만 일반적인 PINN은 입력 데이터 간의 시간적 관계를 충분히 고려하지 못하며, 단순히 좌표 기반의 입력을 사용하는 한계가 있다.
- **적대적 및 대조 학습**: 컴퓨터 비전이나 NLP 분야에서는 데이터 증강과 표현 학습을 위해 널리 사용되나, 저주파 및 고감도 특성을 가진 의료 시계열 데이터 분야에 적용된 사례는 극히 드물다.

### 차별점

PITN은 물리적 제약 조건(Physics Prior)을 유지하면서도, Temporal Block을 통해 시계열의 시간적 특성을 정밀하게 추출한다. 또한, 적대적 샘플링과 대조 학습을 결합하여 단순히 양을 늘리는 것이 아니라, 생리적 의미가 있는 변별적 특징을 학습하도록 설계되었다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

PITN 프레임워크는 크게 **Temporal Block을 통한 특징 추출**, **PINN 기반의 물리적 제약 조건 적용**, 그리고 **적대적 대조 학습(Adversarial Contrastive Learning)** 세 가지 단계로 구성된다.

### 2. Physics-Informed Neural Network (PINN)

본 논문은 혈압의 점진적인 변화 특성을 모델링하기 위해 테일러 근사(Taylor's approximation)를 활용한 물리적 제약 조건을 정의한다. $i$번째 세그먼트 주변의 테일러 다항식 $\tilde{f}_i$는 다음과 같다.

$$\tilde{f}_i(x, u, \theta) = f(x_i, u_i, \theta) + \nabla_{u_i} f(x_i, u_i, \theta)^T (u - u_i)$$

여기서 $\nabla_{u_i} f$는 야코비안(Jacobian) 행렬이다. 네트워크의 예측값 $f$와 테일러 근사값 $\tilde{f}_i$ 사이의 차이를 잔차(Residual) $h_i$로 정의하며, 이를 최소화하는 방향으로 물리 손실 함수 $L_{physics}$를 구성한다.

$$L_{physics} = \frac{1}{N-1} \sum_{i=1}^{N-1} (h_i(x_{i+1}, u_{i+1}, \theta))^2$$

### 3. Temporal Block (시간적 특성 추출)

1차원 생리 신호를 2차원 텐서로 변환하여 다각도에서 특징을 추출하는 구조이다.

- **2D 변환**: FFT(Fast Fourier Transform)를 통해 가장 지배적인 주파수 $f$를 찾아 신호를 $p \times f$ 형태의 2차원 공간으로 재구성(Reshape)한다.
- **Inception Block**: 변환된 2D 텐서에 대해 다양한 크기의 커널을 사용하는 Inception 블록을 적용하여 멀티스케일 시간적 변화를 학습한다.
- **1D 복원**: 추출된 2D 특징을 다시 1D 임베딩으로 변환하여 최종 회귀 헤드(Regression Head)로 전달한다.

### 4. 적대적 훈련 및 대조 학습

- **적대적 샘플 생성**: PGD 알고리즘을 통해 입력 $x$에 미세한 섭동 $\Delta$를 추가하여 $\tilde{x}$를 생성한다. 이때 입력값의 범위를 $\Omega = [\min(x), \max(x)]$로 제한(Clip)하여 생리적 타당성을 유지한다.
- **대조 학습 손실**: 두 샘플 $i, j$의 혈압 값 차이가 임계값 $y_{shift}$보다 작으면 양성 쌍(Positive pair)으로 정의한다.

$$y_{shift} \geq |y_i - y_j|$$

이후 InfoNCE 형태의 손실 함수 $L_{con}$을 사용하여 유사한 혈압을 가진 샘플들의 임베딩을 가깝게 배치한다.

### 5. 최종 학습 목적 함수

전체 모델은 깨끗한 데이터의 손실($L_{clean}$), 적대적 샘플의 손실($L_{adv}$), 대조 학습 손실($L_{con}$), 그리고 물리 손실($L_{physics}$)의 합으로 학습된다.

$$L_{total} = L_{clean} + L_{adv} + L_{con} + \gamma L_{physics}$$

## 📊 Results

### 실험 설정

- **데이터셋**: Graphene-HGCPT (BioZ), Ring-CPT (BioZ), Blumio (PPG, mmWave).
- **평가 지표**: RMSE (Root Mean Square Error), Pearson's correlation coefficient ($r$), ME (Mean Error), SDE (Standard Deviation of Error).
- **학습 조건**: 'Minimal Training Criterion'을 적용하여, 혈압 범위에 따라 매우 적은 수의 샘플만 사용하여 개인화 모델을 학습시켰다.

### 주요 결과

- **정량적 성과**: Graphene-HGCPT 데이터셋에서 `Ours-Full` 모델은 기존 PINN 대비 SBP 기준 상관계수(Correlation)는 31% 향상, RMSE는 14% 감소하는 성과를 보였다.
- **범용성 확인**: BioZ 신호뿐만 아니라 PPG 및 mmWave 신호에서도 기존 SOTA 모델(TimesNet, iTransformer 등)보다 우수한 성능을 기록하였다. 특히 mmWave 데이터셋에서는 TimesNet 대비 SBP 상관계수가 132%나 향상되었다.
- **정성적 분석**: beat-to-beat 혈압 추정 곡선을 분석한 결과, `Ours-Full` 모델이 실제 혈압(Ground Truth)의 변동 추이를 PINN보다 훨씬 정밀하게 추적함을 확인하였다.

### 절제 연구 (Ablation Study)

- **Temporal Block**: PINN 대비 RMSE와 상관계수 모두 크게 개선되어, 시간적 특성 추출이 혈압 추정에 필수적임을 입증하였다.
- **적대적 훈련**: RMSE를 낮추는 데 기여하지만, 상관계수가 일부 하락하는 경향이 발견되었다.
- **대조 학습**: 적대적 훈련으로 인해 하락한 상관계수를 다시 회복시키며, 최종적으로 가장 높은 성능을 달성하게 함을 확인하였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 딥러닝의 데이터 의존성 문제를 해결하기 위해 **'물리 지식(PINN) $\rightarrow$ 시간적 구조 모델링(Temporal Block) $\rightarrow$ 데이터 증강(Adversarial) $\rightarrow$ 표현 학습(Contrastive)'**으로 이어지는 체계적인 파이프라인을 구축하였다. 특히 도메인 지식을 손실 함수에 통합함으로써 적은 데이터로도 높은 개인화 성능을 낼 수 있음을 보였다.

### 한계 및 논의

- **추론 속도**: 1D 신호를 2D로 변환하고 Inception 블록을 통과시키는 과정으로 인해, 단순한 PINN이나 iTransformer보다 추론 시간이 다소 느리다. 하지만 실시간 모니터링 요구 수준 내에서는 수용 가능한 수준이며, 성능 이득이 훨씬 크다고 판단된다.
- **데이터 일관성**: Blumio 데이터셋의 경우 일부 피험자에서 성능이 낮게 나타났는데, 이는 샘플링 포인트의 부족이나 불일치 문제일 가능성이 크며, 추가적인 데이터 확보가 필요하다.

## 📌 TL;DR

본 논문은 데이터 확보가 어려운 커프리스 혈압 추정 문제를 해결하기 위해, 물리 법칙 기반의 제약 조건과 시계열 특성 추출을 결합한 **PITN (Physics-Informed Temporal Network)**를 제안한다. 1D 신호를 2D로 변환하는 Temporal Block과 PGD 기반의 적대적 훈련, 그리고 혈압 유사도 기반의 대조 학습을 통해 **최소한의 학습 데이터만으로도 높은 정밀도의 개인화된 혈압 추정**이 가능함을 입증하였다. 이 연구는 향후 혈당 모니터링 등 다른 의료 시계열 분석 분야로 확장될 가능성이 매우 높다.
