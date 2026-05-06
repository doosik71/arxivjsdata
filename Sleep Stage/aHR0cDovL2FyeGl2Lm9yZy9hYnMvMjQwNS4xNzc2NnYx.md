# SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals

Rahul Thapa et al. (2024)

## 🧩 Problem to Solve

수면 분석의 표준인 수면다원검사(Polysomnography, PSG)는 뇌파(EEG), 심전도(ECG), 호흡 신호 등 다양한 모달리티의 데이터를 수집한다. 전통적으로 이러한 데이터의 분석은 숙련된 기술자의 수동 시각적 검사에 의존해 왔으며, 이는 매우 노동 집약적이고 시간이 많이 소요될 뿐만 아니라 분석가에 따라 오류가 발생할 가능성이 높다.

최근 지도 학습 기반의 딥러닝 모델들이 수면 단계 분류 및 수면 무호흡증(Sleep Disordered Breathing, SDB) 탐지에서 성과를 보이고 있으나, 다음과 같은 한계가 존재한다. 첫째, 대부분의 모델이 좁은 범위의 특정 태스크를 위한 레이블링된 데이터에만 의존한다. 둘째, PSG 센서들이 제공하는 방대한 양의 레이블 없는(unlabeled) 생리학적 역동성을 충분히 활용하지 못한다.

본 논문의 목표는 대규모의 다중 모달 PSG 데이터셋을 활용하여, 다양한 수면 관련 다운스트림 태스크에서 범용적으로 사용할 수 있는 수면 분석용 다중 모달 파운데이션 모델인 SleepFM을 개발하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **다중 모달 대조 학습(Multi-modal Contrastive Learning, CL)**을 통해 뇌 활동 신호(BAS), 심전도(ECG), 호흡 신호 간의 정렬(alignment)을 학습하여 고차원적인 생리학적 표현(representation)을 추출하는 것이다.

특히, 기존의 단순한 쌍별 대조 학습(Pairwise CL)에서 나아가, 특정 모달리티의 임베딩을 나머지 모든 모달리티의 평균 임베딩과 대조하는 **Leave-One-Out CL** 방식을 새롭게 제안하였다. 이를 통해 각 모달리티가 다른 모든 모달리티와 정렬된 통합적인 시맨틱 정보를 캡처하도록 유도하였다.

## 📎 Related Works

기존의 수면 데이터 분석 연구는 주로 Autoencoder, CNN, RNN 기반의 지도 학습 모델에 집중되어 왔다. 이러한 모델들은 수면 단계 측정(Sleep scoring)이나 SDB 탐지에서 유의미한 결과를 냈지만, 데이터셋의 규모가 작고 특정 작업에 과적합되는 경향이 있었다.

한편, 컴퓨터 비전 분야에서는 CLIP과 같은 다중 모달 대조 학습이 큰 성공을 거두었으며, 의료 분야에서도 흉부 엑스레이와 리포트를 정렬하는 ConVIRT 같은 연구가 진행되었다. 하지만 PSG와 같이 뇌, 심장, 폐라는 서로 다른 세 가지 생리학적 시스템에서 발생하는 시계열 신호를 통합적으로 모델링한 다중 모달 파운데이션 모델은 이전까지 거의 연구되지 않았다. SleepFM은 이러한 공백을 메우며, 단순한 지도 학습을 넘어 자가 지도 학습(Self-supervised Learning) 기반의 표현 학습을 수면 분석에 도입했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 데이터셋 및 전처리

본 연구는 1999년부터 2020년까지 스탠포드 수면 클리닉에서 수집된 14,068명의 환자 데이터(총 100,000시간 이상)를 사용한다.

- **모달리티 구성**: 뇌 활동 신호(BAS, 10채널), 심전도(ECG, 2채널), 호흡 신호(Respiratory, 7채널)로 구성된다.
- **전처리**: 모든 신호는 30초 단위의 클립으로 분할되었으며, 샘플링 속도는 256Hz로 통일되었다.

### 2. 임베딩 모델 (Encoder)

각 모달리티의 특성을 추출하기 위해 세 개의 독립적인 1D CNN 인코더를 사용한다. 아키텍처는 EfficientNet을 기반으로 하며, 다음과 같은 구조적 특징을 가진다.

- **Atrous Convolution**: 초기 레이어에서 적용되어 수용 영역(receptive field)을 넓힌다.
- **Inverted Residual Structure**: MobileNetV2의 구조를 채택하여 연산 효율성을 높이고 파라미터 수를 줄였다.
- **Depthwise Separable Convolution**: 모델의 복잡도를 낮추면서 표현 능력을 유지한다.

### 3. 다중 모달 대조 학습 (Multi-modal Contrastive Learning)

SleepFM은 두 가지 대조 학습 프레임워크를 실험하였다.

#### (1) Pairwise CL

두 모달리티 $i$와 $j$ 사이의 임베딩 $x_i^k, x_j^k$ (동일 시간대 클립 $k$)가 서로 가까워지도록 학습한다. 손실 함수는 다음과 같다.
$$l_{\text{pair}}^{i,j,k} = -\log \frac{\exp(\text{sim}(x_i^k, x_j^k) \cdot \exp(\tau))}{\sum_{m=1}^N \exp(\text{sim}(x_i^k, x_j^m) \cdot \exp(\tau))}$$
여기서 $\text{sim}$은 코사인 유사도이며, $\tau$는 학습 가능한 온도(temperature) 파라미터이다.

#### (2) Leave-One-Out (LOO) CL

특정 모달리티 $i$의 임베딩 $x_i^k$를 나머지 모든 모달리티 임베딩의 평균값인 $\bar{x}_{\neq i}^k$와 대조한다.
$$\bar{x}_{\neq i}^k = \text{average}(\{x_j^k \mid j \neq i\})$$
$$l_{\text{LOO}}^{i,k} = -\log \frac{\exp(\text{sim}(x_i^k, \bar{x}_{\neq i}^k) \cdot \exp(\tau))}{\sum_{m=1}^N \exp(\text{sim}(x_i^k, \bar{x}_{\neq i}^m) \cdot \exp(\tau))}$$
이 방식은 각 임베딩이 단일 쌍이 아닌, 전체 모달리티의 통합된 시맨틱을 학습하도록 강제한다.

### 4. 학습 및 추론 절차

1. **Pre-training**: 대규모 레이블 없는 데이터셋으로 CL을 통해 인코더를 학습시킨다.
2. **Embedding Generation**: 학습된 인코더를 고정(freeze)하고, 모든 데이터 클립에 대한 임베딩 벡터를 추출한다.
3. **Downstream Task**: 추출된 임베딩을 입력으로 하여 간단한 로지스틱 회귀(Logistic Regression) 분류기를 학습시켜 수면 단계 및 SDB를 분류한다.

## 📊 Results

### 1. 정량적 성능 평가

SleepFM은 End-to-End로 학습된 CNN 베이스라인과 비교하여 모든 주요 지표에서 압도적인 성능을 보였다.

- **수면 단계 분류 (Sleep Stage Classification)**:
  - SleepFM (LOO): Macro AUROC $0.88$, Macro AUPRC $0.72$
  - Supervised CNN: Macro AUROC $0.72$, Macro AUPRC $0.48$
- **SDB 탐지 (SDB Detection)**:
  - SleepFM (LOO): AUROC $0.85$, AUPRC $0.77$
  - Supervised CNN: AUROC $0.69$, AUPRC $0.61$

### 2. 기타 분석 결과

- **인구통계학적 특성 예측**: 30초의 짧은 클립만으로도 나이와 성별을 높은 정확도로 예측할 수 있었으며, 특히 LOO 방식이 가장 우수한 성능을 보였다.
- **리트리벌(Retrieval) 분석**: 한 모달리티의 임베딩을 통해 다른 모달리티의 대응하는 클립을 찾는 작업에서 무작위 확률보다 500~8000배 높은 성능을 보였다. 다만, 호흡 신호의 가변성이 커서 리트리벌 성능은 상대적으로 낮게 나타났다.
- **Few-shot 평가**: 학습 데이터(환자 수)가 극히 적은 상황에서도 SleepFM은 지도 학습 CNN보다 훨씬 빠르게 성능이 수렴하며 우수한 결과를 보였다.
- **외부 데이터 검증 (External Validation)**: Physionet 2018 챌린지 데이터를 사용하여 검증한 결과, Macro AUROC $0.924$를 기록하며 외부 사이트의 데이터에도 강건하게 일반화됨을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 다중 모달 데이터를 통합하여 학습했을 때 각 개별 모달리티의 표현 능력이 향상됨을 보였다. 특히 Ablation Study를 통해 ECG 신호가 뇌파나 호흡 신호의 임베딩을 풍부하게 만드는 역할을 한다는 점을 발견하였다. 또한, Leave-One-Out CL이 단순 Pairwise CL보다 분류 작업에서 더 우수한 성능을 내는 이유는, 모델이 특정 쌍의 일치보다 전체적인 생리학적 상태의 통합된 표현을 학습했기 때문으로 해석된다.

### 한계 및 비판적 해석

1. **데이터 편향**: 주로 단일 기관(스탠포드)의 데이터로 학습되었으므로, 더 다양한 기관의 데이터를 통한 검증이 필요하다.
2. **모달리티 조합의 미지수**: 어떤 모달리티 조합이 특정 태스크에 가장 최적인지에 대한 정밀한 분석이 부족하며, 이는 향후 연구 과제로 남겨두었다.
3. **현실적 제약**: 모든 PSG 채널이 항상 완벽하게 수집되지 않는 실제 임상 환경에서 일부 채널이 누락되었을 때 모델이 어떻게 작동할지에 대한 논의가 부족하다.

## 📌 TL;DR

SleepFM은 14,000명 이상의 대규모 PSG 데이터를 활용하여 구축된 **최초의 다중 모달 수면 파운데이션 모델**이다. BAS, ECG, 호흡 신호를 통합하는 **Leave-One-Out Contrastive Learning**을 통해 강력한 생리학적 표현을 학습했으며, 이는 수면 단계 분류 및 SDB 탐지에서 기존의 지도 학습 모델을 크게 상회하는 성능을 보였다. 특히 적은 양의 데이터만으로도 높은 성능을 내는 Few-shot 능력과 외부 데이터셋에 대한 뛰어난 일반화 성능을 입증하여, 향후 다양한 임상 환경에서의 수면 분석 자동화에 기여할 가능성이 매우 크다.
