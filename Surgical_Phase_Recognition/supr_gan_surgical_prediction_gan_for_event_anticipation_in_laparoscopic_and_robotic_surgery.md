# SUPR-GAN: SUrgical PRediction GAN for Event Anticipation in Laparoscopic and Robotic Surgery

Yutong Ban, Guy Rosman, Jennifer A. Eckhoff, Thomas M. Ward, Daniel A. Hashimoto, Taisei Kondo, Hidekazu Iwaki, Ozanan R. Meireles, and Daniela Rus (2022)

## 🧩 Problem to Solve

본 논문은 복강경 및 로봇 수술 비디오에서 미래의 수술 단계(surgical phases)를 예측하는 문제를 다룬다. 기존의 수술 AI 연구는 주로 과거의 비디오 프레임을 통해 현재 어떤 수술 단계가 진행 중인지 식별하는 '사후 분석(post-hoc analysis)' 또는 '단계 인식(phase recognition)'에 집중되어 있었다. 그러나 수술 중 발생할 수 있는 합병증을 예방하고 위험을 완화하기 위해서는 단순히 현재 상태를 아는 것을 넘어, 미래에 어떤 이벤트가 발생할지 예측하는 능력이 필수적이다.

특히 담낭 절제술(Laparoscopic Cholecystectomy, LC)의 경우, 총담관 손상(common bile duct injury)과 같은 심각한 합병증이 발생할 확률이 있으며, 이를 방지하기 위해서는 수술자가 즉각적인 조치를 취할 수 있도록 충분한 시간적 여유를 두고 미래 단계를 예측하는 것이 중요하다. 따라서 본 연구의 목표는 과거의 비디오 프레임을 기반으로 미래의 수술 단계 궤적(trajectories)을 확률적으로 샘플링하여 예측하는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 이산적인(discrete) 레이블 시퀀스를 생성할 수 있는 Generative Adversarial Network (GAN) 구조를 수술 단계 예측에 도입한 것이다. 구체적인 기여 사항은 다음과 같다.

- **SUPR-GAN 제안**: 수술 단계와 같은 이산적인 시퀀스를 예측하기 위해 Gumbel-Softmax 층을 결합한 새로운 Discrete GAN 구조인 SUPR-GAN을 제안하였다.
- **다중 궤적 예측**: 단일한 미래 경로만을 예측하는 것이 아니라, 수술 중 발생할 수 있는 다양한 가능성(alternative future trajectories)을 샘플링하여 분포 형태로 예측함으로써 수술의 불확실성을 모델링하였다.
- **통합 프레임워크**: 과거 및 현재 단계의 인식(recognition)과 미래 단계의 예측(prediction)을 하나의 통합된 다중 작업(multi-task) 프레임워크 내에서 수행하도록 설계하였다.
- **임상적 타당성 검증**: 정량적 지표뿐만 아니라, 16명의 전문 외과의를 대상으로 한 설문 조사를 통해 모델이 예측한 경로가 실제 임상적으로 얼마나 타당한지를 정성적으로 평가하였다.

## 📎 Related Works

기존의 수술 워크플로우 분석 연구들은 주로 CNN-LSTM 구조를 사용하여 현재 단계를 인식하는 데 주력하였다. 예를 들어 SV-RCNet과 같은 모델들이 표준적으로 사용되었으며, 일부 연구에서는 긴 시간적 의존성을 캡처하기 위해 통계적 특징을 집계하거나 Temporal Memory Relation Network를 제안하였다. 또한, 수술 도구 식별이나 잔여 수술 시간 예측과 같은 특정 작업에 대한 연구도 존재하였다.

그러나 기존 연구들은 미래의 워크플로우를 예측하는 관점에서는 한계가 있었다. 대부분의 예측 연구는 단순한 회귀 문제(잔여 시간 예측)나 단일한 다음 행동 예측에 그쳤으며, 수술 중 발생하는 복잡하고 비순차적인 단계 전환을 확률적으로 모델링하려는 시도는 부족하였다. 본 논문은 자율 주행 분야의 궤적 예측(trajectory prediction)에서 사용되는 GAN의 개념을 수술 도메인으로 가져와, 연속적인 좌표 값이 아닌 이산적인 단계 레이블 시퀀스를 예측한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

SUPR-GAN은 Encoder-Decoder 기반의 Generator와 이를 판별하는 Discriminator로 구성된다.

1. **Generator Encoder**: 입력된 과거 비디오 프레임 시퀀스 $I$를 처리한다. ResNet을 백본으로 하는 CNN이 각 프레임의 특징을 추출하고, LSTM이 이를 순차적으로 처리하여 최종 상태 벡터 $h_t$를 생성한다. 이 벡터는 현재 수술 단계를 예측하는 `PhaseHead`로 전달되기도 한다.
2. **Generator Decoder**: Encoder의 최종 은닉 상태와 무작위 노이즈 벡터를 결합하여 초기화된다. 이후 LSTM을 통해 미래의 수술 단계 시퀀스를 생성한다. 이때, 연속적인 확률값을 이산적인 레이블로 변환하기 위해 **Gumbel-Softmax** 층을 사용하여 미분 가능한 형태로 이산적 샘플링을 수행한다.
3. **Discriminator**: 과거 시퀀스와 미래 시퀀스를 각각 인코딩하는 두 개의 LSTM 인코더로 구성된다. 최종적으로 생성된 궤적이 실제 데이터에서 샘플링된 것인지(Real), Generator가 만든 것인지(Fake)를 판별하는 이진 분류기 역할을 한다.

### 학습 목표 및 손실 함수

모델은 세 가지 손실 함수의 가중치 합으로 학습된다:
$$L = \omega_1 L_{dis} + \omega_2 L_{rec} + \omega_3 L_{past}$$

- **GAN Loss ($L_{dis}$)**: Discriminator가 진위 여부를 판별하도록 하고, Generator는 Discriminator를 속이도록 학습하는 표준적인 GAN 손실 함수이다.
$$ \min_{D} \max_{G} V(G,D) = \mathbb{E}_{x \sim p_{data}(x)}[\log(D(x))] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z)))] $$
- **Variety Loss ($L_{rec}$)**: Generator가 생성한 $N_s = 10$개의 샘플 중 정답(Ground Truth)과 가장 유사한 샘플과의 거리를 최소화한다. 이는 모델이 정답 근처의 다양한 가능성을 생성하도록 유도한다.
$$ L_{rec}(y, \hat{y}) = \min_{j=1}^{N_s} \sum_{t=t_0+1}^{t_0+T_f} d_L(y^{(j)}_t, \hat{Y}_t) $$
여기서 $d_L$은 이산 레이블 간의 교차 엔트로피(cross-entropy) 거리이다.
- **Past Encoding Loss ($L_{past}$)**: Encoder가 과거 프레임으로부터 현재 단계를 얼마나 정확하게 인식하는지 측정하는 손실 함수로, 일반적인 단계 인식 손실과 동일하다.

### 추론 및 설정

- **Prediction Horizon**: 과거 15초의 영상을 보고 미래 15초의 단계를 예측한다. 15초라는 설정은 임상 전문가들의 의견을 반영한 것으로, 수술자가 상황을 판단하고 조치를 취하기에 적절한 시간이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80(80개 영상, 7개 단계)과 MGH200(200개 영상, 12개 단계)을 사용하였다. MGH200이 더 세밀한 단계 구분과 임상적 변동성을 가지고 있다.
- **비교 대상**: Constant model(마지막 프레임 유지), HMM(Hidden Markov Model), Ours w/o Dis.(Discriminator 없이 Variety loss만 사용한 모델).
- **평가 지표**:
  - **Per-transition accuracy**: 단계가 전환될 때, $\delta=15$초 이내에 새로운 단계를 정확히 예측했는지 측정한다.
  - **Levenshtein distance (LD)**: 예측 시퀀스를 정답 시퀀스로 변환하는 데 필요한 최소 편집 횟수를 측정하며, 값이 낮을수록 정확도가 높다.

### 정량적 결과

- **정확도**: SUPR-GAN은 Cholec80과 MGH200 모두에서 가장 높은 Per-transition accuracy를 기록하였다. 특히 MGH200 데이터셋에서 전체 평균 정확도 53.5%를 달성하여 타 모델 대비 우수함을 보였다.
- **LD 결과**: SUPR-GAN은 전체 샘플 및 전환 구간 모두에서 가장 낮은 Levenshtein distance를 기록하여, 시퀀스 예측의 전반적인 정밀도가 높음을 입증하였다.
- **호라이즌 분석**: 미래 예측 길이($T_f$)가 길어질수록 성능이 하락하며, 과거 정보($T_p$)가 부족할수록 문맥 파악 능력이 떨어져 성능이 저하됨을 확인하였다.

### 정성적 결과 및 외과의 설문

- **궤적 분석**: 시각화 결과, 모델은 단순히 하나의 정답만을 쫓는 것이 아니라, 외과의가 선택할 수 있는 여러 가지 합리적인 대안 경로(alternative trajectories)를 생성해냄을 확인하였다.
- **설문 조사**: 16명의 외과의가 모델의 예측 경로와 실제 경로를 구분하는 테스트를 진행한 결과, 모델이 생성한 경로의 타당성(plausibility)이 매우 높게 나타났다. 흥미로운 점은 외과의들이 직접 미래 단계를 예측한 정확도(53.3%)와 SUPR-GAN의 예측 정확도(53.5%)가 매우 유사하게 나타났다는 것이다.

## 🧠 Insights & Discussion

본 연구는 수술 워크플로우 예측에 있어 GAN의 유연한 샘플링 능력이 매우 효과적임을 보여주었다. 특히 수술 중에는 동일한 상황에서도 수술자의 선호나 환자의 상태에 따라 다른 단계로 진입할 수 있는데, SUPR-GAN은 이러한 '결정 지점(decision-making junction)'에서의 다양성을 모델링할 수 있다는 강점이 있다.

하지만 한계점도 존재한다. 일부 사례에서 모델이 수술의 시간적 순서상 불가능한 단계를 예측하는 경우가 발견되었다. 이는 다양성을 높이기 위한 Variety loss가 때로는 지나치게 이질적인 샘플을 생성하여 노이즈로 작용하기 때문으로 분석된다. 따라서 정확도와 커버리지(coverage) 사이의 적절한 트레이드-오프를 조절하는 추가적인 연구가 필요하다.

결론적으로, 본 모델은 수술 중 위험을 조기에 감지하고 대응할 수 있는 보조 시스템의 기반이 될 수 있으며, 향후 더 복잡한 수술 절차로 확장될 가능성이 크다.

## 📌 TL;DR

본 논문은 복강경 수술 비디오를 통해 미래의 수술 단계를 확률적으로 예측하는 **SUPR-GAN**을 제안한다. 이 모델은 **Discrete GAN** 구조와 **Gumbel-Softmax** 층을 사용하여 수술의 불확실성을 반영한 다양한 미래 궤적을 생성하며, 기존 인식 모델보다 뛰어난 예측 성능과 임상적 타당성을 입증하였다. 이는 향후 수술 중 실시간 위험 완화 및 의사결정 지원 시스템으로 활용될 가능성이 높다.
