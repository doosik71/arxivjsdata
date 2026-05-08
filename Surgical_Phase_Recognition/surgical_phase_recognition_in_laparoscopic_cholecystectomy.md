# Surgical Phase Recognition in Laparoscopic Cholecystectomy

Yunfan Li, Vinayak Shenoy, Prateek Prasanna, I.V. Ramakrishnan, Haibin Ling, and Himanshu Gupta (2022)

## 🧩 Problem to Solve

본 논문은 복강경 담낭 절제술(Laparoscopic Cholecystectomy, LC) 영상에서 수술 단계(Surgical Phase)를 자동으로 인식하는 문제를 다룬다. 수술 워크플로우 분석은 환자의 안전을 개선하고 더 나은 수술 결과를 얻기 위해 로봇 보조 수술 연구에서 매우 중요한 분야이다. 특히, 수술 단계의 자동 분할(Automatic Segmentation)은 수술 교육, 수술 중 보조 및 워크플로우 최적화를 위해 필수적이다.

논문의 주된 목표는 Transformer 기반 모델의 예측 성능을 높이기 위해, 모델의 확신도(Confidence)에 따라 베이스라인 모델과 전이 모델(Transition model) 사이를 동적으로 전환하는 2단계 추론 파이프라인을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 수술 단계가 일반적으로 정해진 순서대로 진행된다는 점에 착안하여, 전체 클래스를 분류하는 모델보다 인접한 두 단계만을 구분하는 2-클래스 분류기(Binary Classifier)가 경계 구간에서 더 높은 정확도를 보인다는 점을 이용하는 것이다.

이를 구현하기 위해 저자들은 다음과 같은 설계를 제안하였다.

1. **Transition Models**: 인접한 두 단계만을 구분하도록 학습된 6개의 2-클래스 분류기를 구축하였다.
2. **Confidence-based Inference**: 베이스라인 모델의 예측 확신도가 낮을 때만 전이 모델로 전환하여 예측하는 전략을 사용한다.
3. **Confidence Calibration**: Transformer 모델 특유의 과잉 확신(Over-confidence) 문제를 해결하기 위해 Temperature Scaling 기법을 적용하여 신뢰할 수 있는 확신도 점수를 생성한다.

## 📎 Related Works

수술 단계 인식 분야의 기존 연구는 다음과 같이 발전해 왔다.

- **통계적 모델**: Conditional Random Field(CRF)와 Hidden Markov Models(HMM)가 초기 단계에서 사용되었으나, 표현 능력이 제한적이며 복잡한 장기 시간적 관계(Long-term temporal relations)를 모델링하는 데 한계가 있었다.
- **순환 신경망(RNN) 및 TCN**: LSTM 네트워크와 이를 ResNet과 결합한 SV-RCNet이 제안되었으며, 이후 pre-computed 공간 특징에서 장기 시간적 관계를 탐색하는 multi-stage TCN 모델인 TeCNO가 제안되었다.
- **Transformer**: 최근에는 Self-attention 메커니즘을 통해 시퀀스 모델링의 패러다임을 바꾼 Transformer 기반 모델들이 등장하였다. Trans-SVNet은 공간 및 시간 임베딩을 결합하여 높은 성능을 보였으며, Opera는 Attention regularization loss를 통해 고품질 프레임에 집중하도록 설계되었다.

본 논문은 이러한 기존 Transformer 기반 방식들이 겪는 **미교정(Miscalibration)** 문제, 즉 모델이 틀린 예측을 하면서도 지나치게 높은 확신도를 갖는 문제를 해결하여 실제 추론 성능을 높이고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 논문은 baseline 모델인 Trans-SVNet을 기반으로 하며, 여기에 전이 모델(Transition models)과 확신도 기반의 추론 전략을 추가한 구조를 가진다.

### 2. Transition Models

전이 모델은 베이스라인 모델과 동일한 방식으로 학습되지만, 오직 인접한 두 단계(예: 1단계 vs 2단계, 2단계 vs 3단계 등)만을 구분하도록 학습된 2-클래스 분류기이다. 총 6개의 전이 모델이 생성되며, 이는 전체 7개 클래스를 한꺼번에 분류하는 것보다 인접 단계 간의 구분 능력이 훨씬 뛰어나다.

### 3. 추론 전략 (Inference Strategies)

저자들은 두 가지 추론 방식을 실험하였다.

- **Transition-based Inference**: 최근 $N$개의 예측값을 저장하는 버퍼 $B_N$을 유지하며, 버퍼 내 다수결 결과가 $i$단계일 경우 $\text{Trans}_{i(i+1)}$ 모델을 사용하여 현재 프레임을 예측한다. 그러나 이 방식은 한 번 잘못된 예측이 나오면 이후 계속 잘못된 전이 모델을 사용하는 **계단식 효과(Cascading effect)**가 발생하여 성능이 저하되는 한계가 있었다.
- **Confidence-based Inference**: 베이스라인 모델의 예측 확신도 $c_{base}$가 설정된 임계값 $t_{conf}$보다 높으면 베이스라인의 예측값을 그대로 사용하고, 낮을 경우에만 이전 예측 단계 $p_{last}$에 기반한 전이 모델 $\text{Trans}_{p_{last}(p_{last}+1)}$을 사용하여 대체 예측값 $p_s$를 생성한다.

### 4. Confidence Calibration

Transformer 모델의 과잉 확신 문제를 해결하기 위해 **Temperature Scaling**을 적용하였다. 이는 학습 후 단일 스칼라 파라미터 $T > 0$를 사용하여 로짓(Logit) 값을 조정하는 방법이다.

수식은 다음과 같다.
$$\hat{q} = \max_{k} \sigma_{SM}(z/T)^{(k)}$$
여기서 $z$는 모델의 로짓 벡터이며, $\sigma_{SM}$은 Softmax 함수이다. $T$ 값을 조정함으로써 확률 분포를 완만하게 만들어 모델의 확신도가 실제 정확도와 일치하도록 교정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Cholec80 (복강경 담낭 절제술 영상 80개)
- **전처리**: 1 fps로 서브샘플링, $250 \times 250$ 픽셀로 리사이즈
- **데이터 분할**: 훈련 40개, 검증 8개, 테스트 32개 영상
- **지표**: Accuracy (%), Negative Log-Likelihood (NLL), Expected Calibration Error (ECE)

### 2. 정량적 결과

**표 1: 2-클래스 전이 모델의 성능**

- 베이스라인(Trans-SVNet)의 정확도는 $87.44\%$인 반면, 각 전이 모델들은 $83.47\% \sim 97.85\%$의 매우 높은 정확도를 보였다.

**표 2: 추론 전략별 성능 비교**

| 모델 | Accuracy (%) |
| :--- | :---: |
| Trans-SVNet (Baseline) | 87.44 |
| Transition-based | 86.92 |
| Confidence-based (w/o calibration) | 65.64 |
| Confidence-based (w/ calibration) | **88.02** |

**표 3: 확신도 교정 결과**

| 모델 | NLL | ECE |
| :--- | :---: | :---: |
| Baseline | 0.576 | 0.215 |
| Calibrated | 0.402 | 0.031 |

교정 후 NLL과 ECE 값이 모두 크게 감소하여, 모델의 확신도가 실제 정답 확률과 훨씬 더 잘 일치하게 되었음을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 단순한 모델 아키텍처의 변경보다 **추론 단계에서의 전략적 보완**과 **확신도 교정(Calibration)**이 실질적인 성능 향상에 기여할 수 있음을 보여주었다.

특히 주목할 점은 다음과 같다.

1. **전이 제약의 양면성**: 국소적인 전이 제약(Local transitional constraints)을 강하게 적용하면 경계에서의 노이즈는 줄일 수 있지만, 한 번의 오답이 이후의 예측을 계속 망치는 'Cascading effect'라는 치명적인 단점이 존재한다.
2. **교정의 필수성**: 확신도 기반의 전환 전략은 모델이 스스로 "언제 틀렸는지"를 정확히 알아야 작동한다. 하지만 최신 딥러닝 모델(특히 Transformer)은 과잉 확신 경향이 강해 교정 없이 전략을 적용했을 때 오히려 성능이 급격히 하락($87.44\% \to 65.64\%$)하는 결과가 나타났다.
3. **결론적 해석**: Temperature Scaling을 통한 교정이 이루어졌을 때 비로소 전이 모델의 높은 정확도를 효과적으로 활용할 수 있었으며, 이는 안전성이 중요한 의료 영상 분석에서 모델의 신뢰도 측정(Confidence estimation)이 얼마나 중요한지를 시사한다.

## 📌 TL;DR

이 논문은 수술 단계 인식에서 Transformer 모델의 과잉 확신 문제를 해결하기 위해 **Temperature Scaling**으로 확신도를 교정하고, 확신도가 낮은 구간에서만 **인접 단계 전용 2-클래스 분류기(Transition Model)**를 사용하는 2단계 추론 파이프라인을 제안하였다. 이를 통해 Cholec80 데이터셋에서 베이스라인보다 향상된 성능($88.02\%$)을 달성하였으며, 이는 향후 수술 워크플로우 분석 및 실시간 수술 보조 시스템의 신뢰성을 높이는 데 중요한 기여를 할 수 있을 것으로 보인다.
