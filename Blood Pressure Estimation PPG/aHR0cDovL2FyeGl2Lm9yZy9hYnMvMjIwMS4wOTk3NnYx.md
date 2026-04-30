# Novel Blood Pressure Waveform Reconstruction from Photoplethysmography using Cycle Generative Adversarial Networks

Milad Asgari Mehrabadi, Seyed Amir Hossein Aqajari, Amir Hosein Afandizadeh Zargari, Nikil Dutt, and Amir M. Rahmani (2022)

## 🧩 Problem to Solve

본 논문은 비침습적인 방법으로 혈압(Blood Pressure, BP)을 지속적으로 모니터링하는 문제를 해결하고자 한다. 일반적으로 수축기 혈압(Systolic Blood Pressure, SBP)과 이완기 혈압(Diastolic Blood Pressure, DBP)을 측정하기 위해 사용되는 커프(cuff) 기반 측정 방식은 일상생활에서 지속적으로 사용하기에 불편함이 크고 현실적이지 않다.

최근 광혈류측정(Photoplethysmography, PPG) 및 심전도(Electrocardiogram, ECG) 신호를 이용해 혈압을 추정하는 딥러닝 기반 접근 방식들이 제안되었으나, 대부분의 기존 연구들은 SBP와 DBP라는 단일 수치(scalar value)를 예측하는 데 그친다는 한계가 있다. 하지만 혈압의 전체 파형(waveform)인 동맥혈압(Ambulatory Blood Pressure, ABP) 신호 자체에는 심혈관 질환의 원인과 심박출량(cardiac volume) 등 풍부한 임상 정보가 포함되어 있다. 따라서 본 논문의 목표는 PPG 신호를 이용하여 전체 ABP 파형을 정밀하게 복원(reconstruction)하는 모델을 개발하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 PPG 도메인에서 ABP 도메인으로의 신호 변환을 수행하기 위해 Cycle Generative Adversarial Networks (CycleGAN)를 도입한 것이다. CycleGAN은 쌍을 이루지 않는(unpaired) 데이터셋 간의 도메인 변환에 강점이 있는 구조로, 이를 통해 PPG 신호의 특성을 유지하면서도 실제 ABP 파형의 분포를 학습하여 정교한 파형 복원을 가능하게 한다. 이를 통해 단순 수치 예측을 넘어 전체 파형을 복원함으로써, 결과적으로 SBP와 DBP 추정 정확도를 기존 연구 대비 최대 2배까지 향상시켰다.

## 📎 Related Works

기존의 혈압 추정 연구는 크게 두 가지 방향으로 진행되었다. 첫 번째는 특징 기반의 통계적 추정 방식이며, 이는 전문가가 정의한 특정 특징(feature)만을 사용하므로 입력 신호의 전체 정보를 충분히 활용하지 못한다는 단점이 있다. 두 번째는 합성곱 층(convolutional layers)을 이용해 신호의 임베딩을 추출하는 딥러닝 기반 방식이다. 이러한 방식들은 SBP와 DBP 수치를 정확하게 예측하지만, ABP 전체 파형을 복원하지 못한다.

ABP 파형 복원을 시도한 일부 선행 연구들은 통계적 방법이나 웨이브렛 신경망(wavelet neural network)을 사용하였으나, 복원 성능이 낮아 임상적 표준을 충족하기에 부족했다. 본 논문은 CycleGAN을 통해 신호 간의 도메인 매핑을 학습함으로써 이러한 한계를 극복하고 복원 정확도를 크게 높였다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
본 연구에서 제안하는 PPG to ABP Translator (PAT)는 PPG 신호를 입력받아 ABP 파형을 생성하는 구조이다. 전체 과정은 다음과 같다:
1. **전처리**: PPG 신호에는 0.1~8Hz 대역 통과 필터(band-pass filter)를, ABP 신호에는 5Hz 저역 통과 필터(low-pass filter)를 적용하여 노이즈를 제거한 뒤 정규화를 수행한다.
2. **윈도우 분할**: 정규화된 신호를 256 샘플 길이의 윈도우로 나누며, 이때 25%의 중첩(overlap)을 둔다.
3. **CycleGAN 학습**: PPG 도메인($X$)과 ABP 도메인($Y$) 사이의 매핑 함수를 학습한다.
4. **평가**: 복원된 ABP 파형에서 SBP와 DBP 수치를 추출하여 실제 값과 비교한다.

### 주요 구성 요소 및 역할
CycleGAN은 두 쌍의 생성기(Generator)와 판별기(Discriminator)로 구성된다.
- **생성기 $G: X \to Y$**: PPG 신호를 ABP 신호로 변환한다.
- **생성기 $F: Y \to X$**: ABP 신호를 다시 PPG 신호로 변환한다.
- **판별기 $D_Y$**: 입력된 ABP 신호가 실제 데이터인지 $G$가 생성한 가짜 데이터인지 판별한다.
- **판별기 $D_X$**: 입력된 PPG 신호가 실제 데이터인지 $F$가 생성한 가짜 데이터인지 판별한다.

### 손실 함수 및 학습 절차
학습의 목표는 실제 데이터의 분포를 맞추는 Adversarial Loss와, 변환 후 다시 원래 도메인으로 돌아왔을 때 원래 신호와 같아야 한다는 Cycle Consistency Loss를 동시에 최적화하는 것이다.

1. **Adversarial Loss**: 생성된 신호가 대상 도메인의 실제 신호와 유사해지도록 유도한다.
$$L^{GAN}(G, D_Y, X, Y) = \mathbb{E}_{y \sim p_{data}(y)}[\log D_Y(y)] + \mathbb{E}_{x \sim p_{data}(x)}[\log(1 - D_Y(G(x)))]$$
(동일한 구조가 $F$와 $D_X$에 대해서도 적용된다.)

2. **Cycle Consistency Loss**: $x \to G(x) \to F(G(x)) \approx x$ 및 $y \to F(y) \to G(F(y)) \approx y$ 관계를 강제하여 개별 입력-출력 간의 일관성을 보장한다.
$$L_{cyc}(G, F) = \mathbb{E}_{x \sim p_{data}(x)}[||F(G(x)) - x||_1] + \mathbb{E}_{y \sim p_{data}(y)}[||G(F(y)) - y||_1]$$

3. **최종 목적 함수**:
$$L(G, F, D_X, D_Y) = L^{GAN}(G, D_Y, X, Y) + L^{GAN}(F, D_X, Y, X) + \lambda L_{cyc}(G, F)$$
여기서 $\lambda$는 사이클 일관성의 중요도를 조절하는 가중치로, 본 논문에서는 10으로 설정하였다.

### 아키텍처 상세
- **Generator**: 2개의 stride-2 convolution, 9개의 residual blocks, 그리고 2개의 fractionally-strided convolutions (stride 0.5)로 구성된다.
- **Discriminator**: $70 \times 70$ PatchGAN 구조를 사용하여 신호의 국소적 특징을 판별한다.

## 📊 Results

### 실험 설정
- **데이터셋**: MIMIC-II 온라인 파형 데이터베이스를 사용하였으며, 92명의 피험자로부터 얻은 5분 분량의 데이터를 활용하였다. (샘플링 주파수 $f_s = 125\text{Hz}$)
- **평가 방법**: 5-fold 교차 검증(cross-validation)을 수행하였으며, 피험자 간(cross-subject) 평가와 피험자 내(per-subject) 평가를 모두 실시하였다.
- **지표**: 평균 절대 오차(MAE), 평균 제곱근 오차(RMSE), Pearson 상관계수($r_P$)를 사용하였으며, 영국 고혈압 학회(BHS) 표준 가이드라인에 따른 등급을 산출하였다.

### 주요 결과
1. **교차 피험자(Cross-subject) 평가**:
   - SBP MAE: $2.89 \pm 4.52\text{mmHg}$
   - DBP MAE: $3.22 \pm 4.67\text{mmHg}$
   - BHS 표준 기준, SBP와 DBP 모두 모든 지표에서 **Grade A**를 달성하였다.

2. **피험자 내(Per-subject) 평가 및 비교**:
   기존의 파형 복원 방식 및 단순 수치 추정 방식과 비교했을 때, 본 모델은 가장 우수한 성능을 보였다.
   - **본 모델**: SBP MAE $2.29\text{mmHg}$, DBP MAE $1.93\text{mmHg}$
   - 기존 파형 복원 방식 $[8]$: SBP MAE $5.9\text{mmHg}$
   - 기존 수치 추정 방식 $[14]$: SBP MAE $3.97\text{mmHg}$
   결과적으로 기존의 최신 기법들보다 혈압 추정 정확도를 최대 2배 가까이 향상시켰다.

## 🧠 Insights & Discussion

본 논문은 CycleGAN을 혈압 추정 분야에 최초로 도입하여, 단순한 수치 예측이 아닌 파형 복원이라는 관점에서 접근함으로써 성능을 크게 개선하였다. 특히, 기존 연구들이 동일 인물의 데이터로 학습하고 테스트하여 일반화 능력을 검증하기 어려웠던 것과 달리, 본 연구는 교차 피험자(cross-subject) 검증을 통해 모델의 일반화 가능성을 입증하였다는 점에서 강점이 있다.

다만, 본 연구는 '깨끗한 PPG 신호(clean PPG signal)'를 전제로 하며, 실제 일상생활에서 발생할 수 있는 심한 움직임 노이즈(motion artifact) 상황에서의 강건성에 대해서는 명확히 다루지 않았다. 또한, CycleGAN의 구조적 특성상 생성된 파형이 시각적으로는 유사하지만, 생리학적 특성(physiological properties)을 완벽하게 반영하고 있는지에 대한 추가적인 분석이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 PPG 신호로부터 ABP 혈압 파형을 복원하기 위해 CycleGAN 기반의 도메인 변환 모델을 제안하였다. 전체 파형을 복원함으로써 단순 수치 예측 모델보다 높은 정확도를 달성하였으며, SBP와 DBP 추정 오차를 획기적으로 줄여 BHS Grade A 기준을 충족하였다. 이 연구는 향후 웨어러블 기기를 통한 비침습적, 연속적 혈압 모니터링 시스템의 실용화를 앞당기는 데 중요한 역할을 할 것으로 기대된다.