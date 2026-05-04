# Semi-supervised Active Learning for Video Action Detection

Ayush Singh, Aayush J Rana, Akash Kumar, Shruti Vyas, Yogesh Singh Rawat (2024)

## 🧩 Problem to Solve

본 논문은 비디오 행동 검출(Video Action Detection, VAD)을 위한 레이블 효율적 학습(label efficient learning) 문제를 해결하고자 한다. 비디오 행동 검출은 단순히 클래스를 분류하는 것을 넘어, 비디오 내에서 행동이 발생하는 시점(temporal)과 위치(spatial)를 동시에 찾아내야 하는 시공간적 국지화(spatio-temporal localization) 작업이 필수적이다.

이러한 특성 때문에 모든 프레임에 대해 시공간적 어노테이션을 수행하는 것은 비용이 매우 많이 들며, 이는 딥러닝 모델 학습에 필요한 대규모 데이터셋 확보의 큰 걸림돌이 된다. 따라서 적은 양의 레이블된 데이터만으로도 높은 성능을 낼 수 있는 효율적인 학습 방법론이 필요하다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 준지도 학습(Semi-supervised Learning, SSL)과 능동 학습(Active Learning, AL)의 상호 보완적 결합을 통해 레이블 효율성을 극대화하는 것이다.

1. **SSL과 AL의 통합 프레임워크**: AL의 고질적인 문제인 '콜드 스타트(cold-start)' 문제(초기 레이블이 너무 적어 모델이 제대로 학습되지 않아 샘플 선택 능력이 떨어지는 현상)를 SSL로 해결하고, SSL의 한계인 '비정보성 샘플 선택' 문제를 AL의 정보성 샘플 선택 기법으로 해결한다.
2. **NoiseAug 제안**: 데이터 기반의 능동 학습을 위해 가우시안 노이즈를 활용한 단순하지만 효과적인 데이터 증강 전략을 제안하여, 모델이 학습 과정에서 겪지 않은 변형을 통해 샘플의 정보성(uncertainty)을 정확히 측정하게 한다.
3. **fft-attention 제안**: 고주파 필터링(High-pass filtering) 기반의 기법을 도입하여, 비디오 내의 배경 소음을 억제하고 행동이 일어나는 영역의 경계(edge)를 강조함으로써 SSL의 의사 레이블(pseudo-label) 생성 품질을 높인다.

## 📎 Related Works

기존의 레이블 효율적 학습 방식은 다음과 같은 한계를 가진다.

- **약지도 학습(Weakly-supervised Learning)**: 비디오 수준의 레이블만 사용하지만, 일반적으로 완전 지도 학습보다 성능이 낮으며 외부에서 사전 학습된 객체 검출기(object detector)에 의존하는 경우가 많다.
- **준지도 학습(Semi-supervised Learning)**: 레이블이 없는 데이터를 활용하지만, 무작위로 선택된 하위 샘플을 사용하기 때문에 정보성이 낮은 샘플이 포함되어 모델 성능이 최적화되지 않을 수 있다. 특히 비디오 데이터의 경우 배경 영역에서 발생하는 노이즈가 의사 레이블의 품질을 저하시킨다.
- **능동 학습(Active Learning)**: 정보성이 높은 샘플을 선택해 레이블링 비용을 줄이려 하지만, 초기 학습 데이터가 극히 적은 경우 모델이 불안정하여 잘못된 샘플을 선택하는 콜드 스타트 문제가 발생한다.

본 논문은 이러한 각 방식의 한계를 SSL의 정규화 능력과 AL의 전략적 샘플 선택 능력을 통합함으로써 극복하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

전체 파이프라인은 **[학습 단계(SSL)] $\rightarrow$ [샘플 선택 단계(AL)] $\rightarrow$ [어노테이션] $\rightarrow$ [다시 학습]**의 순환 구조를 가진다. Student-Teacher 구조의 SSL 프레임워크를 기반으로 하며, AL을 통해 선택된 정보성 높은 샘플들이 학습 데이터셋에 지속적으로 추가된다.

### 2. Active Learning을 위한 NoiseAug 및 샘플 선택

기존의 모델 기반 AL은 정규화(regularization)를 통해 모델을 섭동(perturbation)시키는데, 이는 네트워크에 부정적인 영향을 줄 위험이 있다. 본 논문은 데이터 기반의 **NoiseAug**를 통해 이를 해결한다.

- **NoiseAug**: 원본 샘플 $v$에 가우시안 노이즈 $N_i(0,1)$를 하다마르 곱(Hadamard product) 하여 $R$개의 변형 샘플 $V_\phi$를 생성한다.
  $$V_i^\phi = v_{[T \times H \times W \times C]} \odot N_i(0,1)_{[T \times H \times W \times C]}$$
- **불확실성 측정**: 각 프레임의 픽셀 $p$에 대해 주변 $T$개 프레임의 예측값을 평균 내어 시간적 정규화를 수행한다.
  $$\text{Avg}(f_i[p]) = \frac{1}{T} \sum_{t=i-T/2}^{i+T/2} M(f_t[p]; \theta)$$
- **정보성 점수(Informativeness Score)**: 각 노이즈 변형 샘플들에 대한 불확실성 $U(v)$의 분산(Variance)을 계산하여 최종 점수 $S$를 산출한다.
  $$S = \text{Var}(U(V_1^\phi), U(V_2^\phi), \dots, U(V_R^\phi))$$
  분산이 크다는 것은 모델이 해당 샘플의 노이즈 변형에 민감하게 반응한다는 의미이며, 이는 곧 모델이 해당 샘플을 아직 충분히 학습하지 못한 '정보성 높은 샘플'임을 시사한다.

### 3. Semi-supervised Learning 및 fft-attention

Mean Teacher 프레임워크를 사용하여 Student 모델($M_s$)과 Teacher 모델($M_t$)을 학습시킨다.

- **fft-attention (High-pass Filter)**: 비디오의 배경 영역은 저주파 성분이 많고, 행동 영역의 경계는 고주파 성분이 많다는 점에 착안하여 FFT(Fast Fourier Transform) 기반의 고주파 필터를 적용한다. 이를 통해 행동 영역과 그 경계에 더 높은 가중치 $W$를 부여하는 마스크를 생성한다.
  $$\text{HPF}(f) = \text{FFT}(M(f; \theta))$$
- **일관성 손실 함수(Consistency Loss)**: 단순 MSE 손실 대신, FFT 필터로 생성된 가중치 $W$를 적용하여 행동 영역의 일관성을 더 강하게 강제한다.
  $$\text{FC}(f, f', W) = \frac{1}{x \cdot y} \sum_{p=(1,1)}^{[x,y]} \|M_s(f_p; \theta_s) - M_t(f'_p; \theta_t)\|_2^2 \cdot W_p$$
  최종 일관성 손실은 원본과 변형된 영상 모두에 대해 HPF 가중치를 적용하여 합산한다.
  $$\mathcal{L}_{\text{overall\_cons}} = \lambda_1 \mathcal{L}_{\text{HPF\_cons}} + \lambda_2 \mathcal{L}'_{\text{HPF\_cons}}$$

- **전체 학습 목표**: 레이블된 데이터에 대한 지도 학습 손실($\mathcal{L}_{\text{cls}}, \mathcal{L}_{\text{det}}$)과 레이블되지 않은 데이터에 대한 일관성 손실의 합으로 최적화한다.
  $$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{det}} + \mathcal{L}_{\text{overall\_cons}}$$

## 📊 Results

### 실험 설정

- **데이터셋**: UCF101-24, JHMDB-21 (행동 검출), YouTube-VOS (객체 세그멘테이션 일반화 테스트).
- **지표**: f-mAP (프레임 단위), v-mAP (비디오 단위).
- **백본**: VideoCapsuleNet 사용.

### 주요 결과

1. **성능 향상**: 제안 방법(M-T SSL + AL)은 UCF101-24에서 기존 최고 성능 대비 v-mAP@0.5 기준 +2.4%, JHMDB-21에서 +8.2% 향상된 결과를 보였다.
2. **AL 선택 전략의 우위**: 무작위 선택, MC Uncertainty, MC Entropy 기반 선택보다 제안한 NoiseAug 기반 선택이 모든 데이터셋에서 일관되게 높은 성능을 기록하였다. 특히 MC Entropy는 초기 단계에서 콜드 스타트 문제로 인해 성능이 저조했다.
3. **SSL 구성 요소의 효과**: Table 2의 절제 연구(Ablation Study)를 통해, 단순 일관성 학습보다 Mean Teacher 구조가 우수하며, 특히 FFT 필터를 적용했을 때 성능이 크게 향상됨을 확인하였다.
4. **일반화 가능성**: YouTube-VOS 데이터셋을 이용한 비디오 객체 세그멘테이션 작업에서도 무작위 선택이나 PI-consistency보다 높은 성능을 보여, 제안 방법이 다른 밀집 예측(dense prediction) 작업에도 적용 가능함을 입증하였다.

## 🧠 Insights & Discussion

### FFT 필터의 역할에 대한 해석

논문은 모델의 예측 상태를 네 가지(신뢰도/일관성 조합)로 구분하여 분석한다. 단순 일관성 손실은 신뢰도가 낮고 일관성도 낮은(LCF-LC) 영역까지 포함하여 학습하려 하지만, 이는 정답 데이터가 부족한 상황에서 노이즈만 학습시킬 위험이 있다. FFT 기반의 고주파 필터는 신뢰도는 높지만 일관성이 낮은(HCF-LC) 영역, 즉 모델이 확신은 하지만 경계 부분에서 갈팡질팡하는 영역에 가중치를 줌으로써 학습의 효율성을 높인다.

### NoiseAug의 강점

기존 SSL에서 사용되는 Strong/Weak Augmentation은 모델이 이미 학습 과정에서 경험한 변형들이므로, AL 단계에서 이를 사용하면 모델이 이미 내성이 생겨 불확실성을 낮게 측정할 가능성이 크다. 반면 NoiseAug는 학습 시 보지 못한 새로운 형태의 섭동을 제공함으로써, 모델이 실제로 취약한 샘플을 더 정확하게 식별하게 한다.

### 한계 및 논의

- FFT 필터의 반경(radius) 설정에 따라 성능이 달라질 수 있으며, 이에 대한 민감도 분석을 수행하여 어느 정도의 강건성을 확인하였다.
- 비디오 데이터의 특성상 시간적 일관성을 위해 $\text{Avg}$ 함수를 도입하였으나, 더 복잡한 시간적 관계를 모델링하는 방법론에 대한 추가 연구가 필요할 수 있다.

## 📌 TL;DR

본 논문은 비디오 행동 검출을 위해 **준지도 학습(SSL)과 능동 학습(AL)을 통합한 프레임워크**를 제안한다. 특히 **NoiseAug**를 통해 정보성 높은 샘플을 정밀하게 선택하고, **fft-attention(고주파 필터)**를 통해 행동 영역의 경계를 강조함으로써 의사 레이블의 품질을 높였다. 실험 결과, 기존의 약지도/준지도 학습 방식보다 뛰어난 성능을 보였으며, 다른 비디오 밀집 예측 작업으로의 확장 가능성도 확인하였다. 이 연구는 레이블 비용이 매우 높은 비디오 분석 분야에서 실질적인 데이터 효율성을 달성하는 데 중요한 기여를 한다.
