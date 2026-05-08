# Surgical Temporal Action-aware Network with Sequence Regularization for Phase Recognition

Zhen Chen, Yuhao Zhai, Jun Zhang, Jinqiao Wang (2023)

## 🧩 Problem to Solve

본 논문은 수술실에서 외과의를 보조하기 위한 컴퓨터 보조 수술 시스템의 핵심 요소인 수술 단계 인식(Surgical Phase Recognition) 문제를 해결하고자 한다. 수술 단계 인식은 수술 비디오에 대한 종합적인 이해를 필요로 하며, 이는 수술 절차 모니터링, 수술 일정 관리, 팀 협동 촉진 및 초보 외과의 교육 등에 매우 중요하다.

기존의 연구들은 상당한 진전을 이루었으나, 두 가지 주요한 한계점을 가지고 있다. 첫째, 계산 자원 소비를 줄이기 위해 프레임별 시각적 특징(Frame-wise visual features)을 2D 네트워크로 추출하는데, 이 과정에서 수술 동작(Surgical action)의 공간적, 시간적 지식이 무시되어 이후의 프레임 간 모델링(Inter-frame modeling) 성능을 저하시킨다. 둘째, 대부분의 기존 방식들이 원-핫(One-hot) 단계 레이블을 이용한 단순한 분류 손실 함수(Classification loss)만을 사용하여 네트워크를 최적화하므로, 감독 신호(Supervision)가 부족하여 모델이 오버피팅(Over-fitting)되기 쉽다는 문제가 있다. 따라서 본 논문의 목표는 수술 동작의 시공간적 정보를 효율적으로 추출하고, 효과적인 정규화(Regularization) 기법을 통해 수술 단계 인식의 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시각적 특징 추출 단계에서 수술 동작의 다중 척도 시간적 정보를 통합하고, 보조 분류기를 통한 상호 정규화 기법을 도입하는 것이다.

이를 위해 저자들은 **MS-STA(Multi-Scale Surgical Temporal Action)** 모듈과 **DSR(Dual-classifier Sequence Regularization)** 기법을 제안하였다. MS-STA는 2D 네트워크의 비용으로 3D 네트워크와 유사한 시간적 동작 정보를 캡처하며, DSR은 용량이 작은 보조 분류기를 도입하여 학습 초기와 후기 단계에서 주 분류기와 서로를 정규화함으로써 모델의 일반화 성능을 높인다.

## 📎 Related Works

수술 단계 인식 분야의 초기 연구들은 이 문제를 단순한 프레임별 분류 문제로 정의하고 수술 도구의 보조 주석(Auxiliary annotations)을 활용한 멀티태스크 학습을 수행하였다. 이후 3D 컨볼루션(3D Convolutions)을 사용하여 시간적 지식을 캡처하려는 시도가 있었으나, 이는 매우 높은 계산 자원을 소모한다는 단점이 있었다.

이를 극복하기 위해 최근의 주류 방법론들은 2D CNN으로 각 프레임의 특징 벡터를 먼저 추출한 뒤, LSTM, Temporal Convolutions, 또는 Transformer를 사용하여 프레임 간의 시간적 관계를 집계하는 다단계 파이프라인을 채택하고 있다. 그러나 이러한 방식들은 여전히 2D 네트워크가 프레임을 특징 벡터로 변환하는 과정에서 시공간적 정보가 손실된다는 점과, 단순한 분류 손실 함수만으로는 수술 비디오의 복잡한 지식을 충분히 학습시키기에 감독 정보가 부족하다는 한계를 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

STAR-Net은 입력 비디오 시퀀스 $\{x_{n-t}\}_{t=0}^{T-1}$를 받아 현재 프레임 $x_n$의 수술 단계를 예측한다. 전체 파이프라인은 **[2D CNN + MS-STA] $\rightarrow$ [Transformer (Spatial & Temporal Attention)] $\rightarrow$ [Classifiers (Task & Auxiliary)]** 순으로 구성된다.

### 2. Multi-Scale Surgical Temporal Action (MS-STA)

MS-STA는 2D 백본 네트워크 내에 삽입되어 시각적 특징 $f \in \mathbb{R}^{T \times H \times W \times D}$에 수술 동작의 다중 척도 시간 정보를 통합한다.

- **Temporal Difference (TDiff) 연산**: 인접한 두 프레임 사이의 수술 동작을 캡처하기 위해 제안되었다. 입력 특징 $f$를 시간 축으로 한 프레임 이동시킨 지연 특징 $D(f, 1)$을 생성하고, 이를 입력 특징에서 요소별로 뺄셈하여 동작 특징 $a_1$을 계산한다.
  $$a_1 = M(f - D(f, 1))$$
  여기서 $M(\cdot)$은 첫 번째 프레임의 뺄셈 결과를 0으로 설정하는 액션 마스크(Action mask)이다.

- **다중 척도 통합**: TDiff 연산을 반복 수행하여 더 긴 시간 범위의 동작 특징 $a_2, \dots, a_\tau$를 생성한다. 이후 $\tau$개의 서로 다른 시간 척도를 가진 특징들을 연결(Concatenate)하고, 단일 3D 컨볼루션 층을 통해 통합된 동작 특징 $a_{ms}$를 추출한다.
  $$a_{ms} = W \circledast [a_1, a_2, \dots, a_\tau]$$
  최종적으로 $a_{ms}$를 입력 특징 $f$에 더하는 잔차 학습(Residual learning) 방식을 통해 각 프레임에 수술 동작 지식을 부여한다.

### 3. Dual-Classifier Sequence Regularization (DSR)

DSR은 주 분류기(Task Classifier)와 용량이 작은 보조 분류기(Auxiliary Classifier) 간의 상호 정규화를 통해 오버피팅을 방지한다.

- **분류기 구성**: 주 분류기는 Transformer를 거친 토큰을 사용하여 예측값 $p_{task}$를 생성하며, 보조 분류기는 2D 백본의 특징에 Spatial GAP(Global Average Pooling)를 적용하여 예측값 $p_{aux}$를 생성한다.
- **상호 정규화 전략**:
  - **초기 시퀀스($E$)**: Transformer 이후의 주 분류기는 이전 프레임 정보가 부족하여 불안정하므로, 보조 분류기가 주 분류기를 정규화한다.
  - **후기 시퀀스($L$)**: 보조 분류기는 장기적인 맥락 정보가 부족하므로, 주 공부된 주 분류기가 보조 분류기를 정규화한다.
- **손실 함수**: KL 발산(Kullback-Leibler divergence)을 사용하여 다음과 같이 정의한다.
  $$\mathcal{L}_{DSR} = \sum_{i \in E} KL(p_{task}^{(i)} || \hat{p}_{aux}^{(i)}) + \sum_{j \in L} KL(p_{aux}^{(j)} || \hat{p}_{task}^{(j)})$$
  여기서 $\hat{p}$는 그래디언트 전파를 차단한 상수값을 의미한다.

### 4. 학습 및 추론 절차

학습은 2단계로 진행된다. 먼저 2D 백본과 MS-STA를 교차 엔트로피 손실($\mathcal{L}_{CE}$)로 학습시키고, 이후 Transformer와 두 분류기를 DSR 손실을 포함하여 학습시킨다. 전체 손실 함수는 다음과 같다.
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{DSR}$$
추론 시에는 학습된 STAR-Net을 통해 엔드-투-엔드 방식으로 온라인 프레임별 예측을 수행한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**:
  - **Gastrectomy Phase Dataset**: 100개의 위절제술 비디오 (8단계로 구성, 자체 수집).
  - **Cholec80**: 80개의 담낭절제술 비디오 (7단계로 구성, 공개 데이터셋).
- **평가 지표**: Accuracy (AC), Precision (PR), Recall (RE), Jaccard Index (JA)를 사용하였다.
- **구현 세부사항**: ResNet-18을 백본으로 사용하였으며, 시간 척도 $\tau=5$, 시퀀스 길이 $T=20$으로 설정하였다.

### 2. 주요 결과

- **위절제술 데이터셋**: STAR-Net은 AC 89.2%, JA 73.5%를 기록하며 Trans-SVNet 등 기존 SOTA 모델들을 상당한 차이로 앞질렀다. t-test 결과 $P\text{-value} < 1 \times 10^{-5}$로 통계적 유의성을 확보하였다.
- **Cholec80 데이터셋**: AC 91.2%, PR 91.6%, JA 79.5%를 달성하여 최우수 성능을 보였다. 특히 기존 모델들에 비해 파라미터 수와 연산량(FLOPs) 측면에서 매우 효율적임이 입증되었다.
- **Ablation Study**: MS-STA만 제거했을 때 AC가 2.3%p 하락하고, DSR만 제거했을 때 1.3%p 하락하여, 두 모듈 모두 성능 향상에 필수적임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 3D 컨볼루션의 무거운 연산 비용을 피하면서도, TDiff 연산과 단일 3D Conv 층을 통해 효율적으로 시간적 동작 정보를 추출할 수 있음을 보여주었다. 특히 시각화 결과(Fig 5)를 통해 초음파 칼, 그라스퍼, 훅과 같은 수술 도구의 움직임이 MS-STA에 의해 잘 포착되고 있음을 확인하였다.

또한, 단순히 정답 레이블만을 쫓는 것이 아니라, 학습 시퀀스의 위치(초기 vs 후기)에 따라 서로 다른 분류기가 서로를 가이드하게 하는 DSR 기법이 수술 비디오와 같은 데이터 부족 환경에서 오버피팅을 억제하는 강력한 도구가 될 수 있음을 시사한다. 다만, 논문 내 Fig 3에서 언급되었듯 수술 단계별 데이터 불균형(Class Imbalance) 문제가 여전히 존재하며, 이는 향후 해결해야 할 과제로 보인다.

## 📌 TL;DR

STAR-Net은 수술 단계 인식의 정확도를 높이기 위해 **다중 척도 수술 동작 추출 모듈(MS-STA)**과 **이중 분류기 시퀀스 정규화(DSR)**를 제안한 모델이다. MS-STA는 2D 네트워크의 효율성을 유지하면서 시공간적 동작 정보를 캡처하며, DSR은 보조 분류기를 통해 학습 과정을 정규화한다. 위절제술 및 담낭절제술 데이터셋에서 SOTA 성능과 높은 연산 효율성을 동시에 달성하였으며, 이는 향후 실시간 수술 보조 시스템 구축에 중요한 기여를 할 것으로 기대된다.
