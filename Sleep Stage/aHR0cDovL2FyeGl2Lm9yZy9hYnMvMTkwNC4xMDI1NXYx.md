# End-to-end Sleep Staging with Raw Single Channel EEG using Deep Residual ConvNets

Ahmed Imtiaz Humayun, Asif Shahriyar Sushmit, Taufiq Hasan, and Mohammed Imamul Hassan Bhuiyan (2019)

## 🧩 Problem to Solve

본 논문은 뇌파(EEG) 신호를 이용한 자동 수면 단계 분류(Automatic Sleep Stage Classification, ASSC) 문제를 해결하고자 한다. 수면 장애는 전 세계적으로 증가하는 보건 문제이며, 이를 정확히 진단하는 것은 전반적인 건강 유지에 필수적이다.

전통적으로 수면 단계는 다원수면검사(Polysomnography, PSG)를 통해 EEG, EOG, ECG 등 여러 생체 신호를 종합하여 전문가가 판독한다. 그러나 이러한 수동 판독 과정은 시간이 많이 소요되며 전문가의 주관이 개입될 수 있다. 기존의 자동화 시도들은 주로 수작업으로 설계된 특징 추출(Hand-engineered feature extraction) 방식이나 얕은 딥러닝 모델을 사용했다. 수작업 특징 추출 방식은 특정 인구 집단에 최적화되어 일반화 성능이 떨어지며, 어떤 특징이 가장 변별력이 있는지에 대한 합의가 부족하다는 한계가 있다. 따라서 본 논문의 목표는 단일 채널 raw EEG 신호만을 입력으로 하여, 전처리나 수동 특징 추출 없이 수면 단계를 직접 분류하는 고성능의 end-to-end 딥러닝 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 매우 깊은 층의 1차원 합성곱 신경망(1DCNN)을 구축하되, 학습 과정에서 발생하는 기울기 소실(Vanishing Gradient) 문제를 해결하기 위해 **Residual Architecture(ResNet)**를 도입하는 것이다.

기존의 수면 단계 분류 연구들이 4~12층 정도의 상대적으로 얕은 모델을 사용한 것과 달리, 본 연구에서는 34층의 깊은 1DCNN 구조를 제안한다. Residual Block의 skip-connection을 통해 모델이 항등 함수(Identity operation)를 근사할 수 있게 함으로써, 네트워크의 깊이를 획기적으로 늘리면서도 최적화 및 수렴 속도를 개선하고 더 복잡하고 일반화된 특징을 학습할 수 있도록 설계하였다.

## 📎 Related Works

수면 단계 분류를 위한 기존 연구들은 크게 세 가지 방향으로 진행되었다.

1. **특징 기반 머신러닝**: IIR 필터, Wavelet Transform 등을 통해 시간-주파수 도메인의 특징을 추출하고 SVM이나 Decision Tree와 같은 알고리즘을 사용하는 방식이다. 이러한 방법은 도메인 지식에 의존하며 일반화 능력이 부족하다.
2. **얕은 딥러닝 모델**: 1DCNN, Bi-directional LSTM 등을 사용한 end-to-end 시스템이 제안되었으나, 층의 깊이가 낮아 복잡한 신호 패턴을 충분히 학습하는 데 한계가 있었다.
3. **2D 이미지 변환 방식**: EEG 신호를 스펙트럼 분해를 통해 2D 이미지로 변환한 후 VGG16과 같은 2DCNN을 적용하는 방식이 존재한다.

본 논문은 이러한 기존 방식들과 달리, raw 1D 신호를 그대로 사용하면서 ResNet 구조를 통해 모델의 깊이를 극대화함으로써 성능을 높였다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 모델은 30초 길이의 raw 단일 채널(Fpz-Cz) EEG 신호를 입력받아 수면 단계(5단계 또는 6단계)를 예측하는 end-to-end 1DCNN 구조이다. 전체 네트워크는 34개의 1DCNN 층으로 구성되어 있으며, Depth가 깊어질수록 커널의 개수가 증가하는 구조를 가진다.

### 2. 주요 구성 요소 및 알고리즘

- **Residual Block**: 본 모델의 핵심 단위로, 두 가지 형태가 존재한다.
  - **Pooling이 없는 블록**: $\text{CONV} \rightarrow \text{ReLU} \rightarrow \text{BN} \rightarrow \text{CONV} \rightarrow \text{ReLU} \rightarrow \text{BN}$ 순으로 구성되며, 입력 텐서를 출력에 직접 더하는 skip-connection이 적용된다.
  - **Pooling이 있는 블록**: 메인 경로에 Max-pooling이 포함되며, skip-connection 경로에도 텐서의 깊이를 맞추기 위해 커널 크기가 1인 $\text{CONV}$ 층이 추가된다.
- **Pre-activation Design**: 활성화 함수를 적용하기 전에 입력 텐서와 변환된 출력을 먼저 더하는 방식을 채택하여 기울기가 하위 층으로 더 원활하게 흐르도록 설계하였다.
- **하이퍼파라미터**:
  - 모든 $\text{CONV}$ 층의 커널 크기는 16으로 고정된다.
  - 필터의 개수는 $64 \times k$ (단, $k \in \{1, 2, 3, 4\}$)이며, 매 8개 층마다 $k$가 1씩 증가한다.
  - Max-pooling은 데이터 크기를 2배로 줄이는(sub-sampling) 역할을 수행한다.

### 3. 학습 절차 및 손실 함수

- **손실 함수**: 클래스 불균형(Class Imbalance) 문제를 해결하기 위해, 훈련 데이터 내 각 클래스의 빈도에 따라 가중치를 부여한 **Weighted Cross-entropy Loss**를 사용한다.
- **최적화**: Adam Optimizer를 사용하였으며, 초기 학습률(Learning Rate)은 $0.001$이다. 매 10 에포크마다 학습률을 10분의 1로 감소시키는 스케줄링을 적용하였다.
- **데이터 증강**: 일반화 성능을 높이기 위해 훈련 데이터에 대해 Rolling shifts(신호를 약간씩 밀어서 생성)를 수행하였다.
- **정규화**: 과적합 방지를 위해 post-activation 확률 0.5의 Dropout을 적용하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: PhysioNet Sleep-EDF Expanded Database (총 42명의 피험자, 61개의 기록).
- **평가 태스크**:
  - **RS-task (Random Split)**: SC(Sleep Cassette)와 ST(Sleep Telemetry) 데이터 전체를 대상으로 피험자 독립적(Patient Independent)으로 71%:29% 분할하여 테스트.
  - **SC-task (Sleep Cassette)**: SC 서브셋만 사용하여 피험자 독립적으로 70%:30% 분할하여 테스트.
- **평가 지표**: Epoch-wise Accuracy(에포크별 정확도), Patient-wise Accuracy(환자별 정확도), Sensitivity, Specificity.

### 2. 주요 결과

- **SOTA 비교**: 기존의 환자 독립적 테스트 결과들과 비교했을 때, 제안된 모델은 매우 경쟁력 있는 성능을 보였다.
- **직접 비교 (vs. Aboalayon et al.)**: 가장 우수한 성능을 보였던 Aboalayon et al.의 IIR-MMD-DT 방식과 동일한 조건에서 비교 실험을 수행한 결과:
  - **SC-task**: Epoch-wise Accuracy 기준 약 6.8%의 상대적 향상을 보였다.
  - **RS-task**: Epoch-wise Accuracy 기준 약 6.3%의 상대적 향상을 보였다.
  - **Patient-wise Accuracy**: SC-task에서 6.76%, RS-task에서 6.45%의 향상을 기록하였다.

## 🧠 Insights & Discussion

### 1. 데이터 분할 방식의 중요성

저자들은 기존 연구들 중 상당수가 동일 환자의 데이터가 훈련 세트와 테스트 세트에 동시에 포함되는 'Example Splitting' 방식을 사용하여 성능이 과도하게 높게 측정되었음을 지적한다. 본 연구는 **Patient Independent Split**을 엄격히 적용함으로써 실제 임상 적용 가능성에 더 가까운 강건한 평가를 수행하였다.

### 2. 데이터 소스 간의 이질성 (Heterogeneity)

실험 결과, RS-task의 성능이 SC-task보다 낮게 나타났으며, 특히 ST(Sleep Telemetry) 데이터에서 정확도가 떨어지는 경향이 관찰되었다. 이를 분석하기 위해 ANOVA 테스트를 수행한 결과, $\gamma$-band EnergySis 특징을 제외한 대부분의 시간-주파수 특징에서 SC와 ST 데이터 간의 분포 차이가 통계적으로 유의미함($p < 0.001$)이 밝혀졌다. 즉, 병원(ST)과 가정(SC)에서 수집된 데이터의 특성이 서로 다르기 때문에, 두 소스가 섞인 RS-task에서 성능 저하가 발생한 것으로 해석할 수 있다.

### 3. 비판적 해석

본 모델은 깊은 구조를 통해 높은 성능을 달성했지만, 34층이라는 깊은 층이 수면 단계 분류라는 특정 태스크에서 반드시 필요한지에 대한 Ablation Study(절제 연구)가 부족하다. 또한, 1D-ResNet이 구체적으로 어떤 신호 패턴을 학습하여 성능을 높였는지에 대한 시각적 분석이나 설명 가능성(Explainability)에 대한 논의가 부족한 점이 아쉽다.

## 📌 TL;DR

본 논문은 raw 단일 채널 EEG 신호를 이용하여 수면 단계를 자동 분류하는 **34층 규모의 깊은 1D-ResNet 모델**을 제안한다. 수작업 특징 추출 없이 end-to-end 학습을 수행하며, 특히 피험자 독립적(Patient Independent) 평가 방식을 통해 기존 SOTA 모델 대비 Epoch-wise 정확도를 약 6.3%~6.8% 향상시켰다. 또한, 데이터 수집 환경(가정 vs 병원)에 따른 신호의 이질성이 모델 성능에 영향을 미친다는 점을 분석하였다. 이 연구는 향후 웨어러블 기기를 이용한 실시간 수면 모니터링 시스템 구축에 중요한 기초가 될 수 있다.
