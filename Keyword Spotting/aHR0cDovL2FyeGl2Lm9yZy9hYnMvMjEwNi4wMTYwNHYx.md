# Noisy student-teacher training for robust keyword spotting

Hyun-Jin Park, Pai Zhu, Niranjan Subrahmanya, Ignacio Lopez Moreno (2021)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 Streaming Keyword Spotting(KWS) 시스템의 강건성(Robustness) 향상이다. 일반적인 KWS 모델은 지도 학습(Supervised Learning)에 의존하는데, 이는 대규모의 고품질 레이블 데이터(Labeled Data)를 확보하는 데 많은 비용과 시간이 소요된다는 한계가 있다.

또한, 데이터 증강(Data Augmentation) 기법 중 하나인 Spectral Augmentation(SpecAugment)은 자동 음성 인식(ASR) 분야에서 효과적임이 증명되었으나, KWS와 같은 이진 분류(Binary Classification) 문제에 그대로 적용할 경우 성능이 저하되는 문제가 발생한다. 이는 매우 공격적인 데이터 증강이 양성(Positive) 샘플을 음성(Negative) 샘플처럼 보이게 만들 수 있는데, 지도 학습의 하드 레이블(Hard-label, $\in \{0, 1\}$)은 이러한 입력 패턴의 변화를 반영하지 못하고 강제로 양성 레이블을 부여함으로써 False Accept Rate를 높이기 때문이다.

따라서 본 논문의 목표는 대규모의 미분류 데이터(Unlabeled Data)를 활용하고, 공격적인 데이터 증강을 적용하면서도 모델의 성능을 저하시키지 않는 Noisy Student-Teacher 학습 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **교사(Teacher) 모델과 학생(Student) 모델 모두에게 동일하게 공격적인 데이터 증강을 적용하는 Self-training 구조**를 설계한 것이다.

기존의 Noisy Student 방식은 교사 모델에는 깨끗한 데이터를 제공하고 학생 모델에만 노이즈가 섞인 데이터를 제공했다. 하지만 본 연구에서는 KWS가 양성 패턴의 공간이 매우 좁은 불균형한 이진 분류 문제라는 점에 주목하여, 교사 모델 또한 증강된 데이터를 입력받게 함으로써 입력 데이터의 훼손 정도에 따라 동적으로 변화하는 **소프트 레이블(Soft-label, $\in [0, 1]$)**을 생성하도록 하였다. 이를 통해 과도한 증강으로 인해 데이터의 성격이 변하더라도 교사 모델이 낮은 신뢰도의 레이블을 제공함으로써 학생 모델이 잘못된 정보를 학습하는 것을 방지할 수 있다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개한다.

- **Semi-supervised Learning**: 소량의 레이블 데이터와 대량의 미분류 데이터를 함께 사용하여 인접한 데이터의 활성화를 예측하는 방식으로 특징을 학습하며, 이후 분류 네트워크를 학습시킨다.
- **Self-training**: 교사 네트워크가 생성한 의사 레이블(Pseudo-labels)을 사용하여 학생 네트워크를 학습시키며, 이 과정을 반복적으로 수행하여 성능을 높인다.
- **Data Augmentation**: 잔향(Reverberation) 추가나 노이즈 믹싱과 같은 전통적 방법과 더불어, 주파수 빈(Frequency bins)이나 시간 프레임을 무작위로 마스킹하는 SpecAugment가 최근 ASR 성능 향상에 기여하였다.

기존 Noisy Student 방식($[18, 25]$)과의 차별점은, 기존 방식이 다중 클래스 분류(ImageNet, ASR 등)에 최적화되어 학생에게만 노이즈를 주었던 것과 달리, 본 연구는 KWS의 특성에 맞춰 교사와 학생 모두에게 동일한 노이즈를 부여하고 레이블 데이터와 미분류 데이터 모두를 교사 모델에 통과시켜 소프트 레이블을 생성한다는 점이다.

## 🛠️ Methodology

### 전체 파이프라인 및 학습 절차

학습 과정은 크게 두 단계로 구성된다.

1. **1단계 (Teacher 학습)**: 레이블 데이터($L$)와 전통적인 데이터 증강(잔향 및 배경 소음 추가)을 사용하여 기본 교사 모델($T_0$)을 지도 학습 방식으로 학습시킨다.
2. **2단계 (Student 학습)**: 1단계에서 학습된 교사 모델을 이용하여 학생 모델($S_k$)을 학습시킨다. 이때 레이블 데이터와 미분류 데이터의 합집합($L \cup U$)에 공격적인 SpecAugment를 적용한다. 교사 모델은 이 증강된 데이터를 입력받아 소프트 레이블($y^T$)을 생성하고, 학생 모델은 이를 정답으로 삼아 학습한다. 학습이 완료된 학생 모델은 다음 반복 단계의 교사 모델($T_{k+1}$)이 된다.

### 손실 함수 및 방정식

학생 모델의 학습을 위해 Cross Entropy(CE) 손실 함수를 사용한다. 본 모델은 Encoder와 Decoder의 두 가지 출력을 가지므로, 각각에 대한 손실을 가중 합산하여 최종 손실을 계산한다.

전체 손실 함수는 다음과 같이 정의된다.
$$\text{Loss}_{\text{student-teacher}} = \alpha \cdot \text{Loss}_E + \text{Loss}_D$$

여기서 $\text{Loss}_E$와 $\text{Loss}_D$는 각각 Encoder와 Decoder의 Cross Entropy 손실이다.
$$\text{Loss}_D = \text{crossentropy}(y^T_d, y^S_d)$$
$$\text{Loss}_E = \text{crossentropy}(y^T_e, y^S_e)$$

입력 데이터 $x$에 증강($\text{augment}$)을 적용한 후, 교사와 학생 모델의 출력(소프트 레이블)은 다음과 같다.
$$y^T = [y^T_d, y^T_e] = f^T(\text{augment}(x))$$
$$y^S = [y^S_d, y^S_e] = f^S(\text{augment}(x))$$

### SpecAugment 및 모델 아키텍처

- **SpecAugment**: 주파수 빈(Frequency bins)이나 시간 프레임을 덩어리(Chunks) 단위로 완전히 마스킹하는 공격적인 증강 기법이다.
- **모델 아키텍처**: 교사와 학생 모델 모두 동일한 구조를 사용한다. 7개의 단순화된 Convolution Layer와 3개의 Projection Layer로 구성되며, Encoder-Decoder 구조를 띈다. Encoder는 40차원의 스펙트럼 주파수 에너지 벡터를 입력받아 음소와 유사한 단위로 인코딩하고, Decoder는 이를 바탕으로 키워드의 존재 여부를 이진 출력한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 지도 학습용 데이터 250만 개, 미분류 데이터 1,000만 개를 사용하였다.
- **평가 데이터셋**: Near-field Clean, Far-field, Query Logs(QLog) 등 다양한 환경을 대표하는 8개의 평가 셋을 사용하였다.
- **지표**: $0.1 \text{ FA/h}$ (시간당 오경보 0.1회) 기준에서 False Rejection(FR) rate를 측정하여 비교하였다. FR rate가 낮을수록 성능이 우수함을 의미한다.

### 주요 결과

실험 결과, 제안된 $\text{ST+sAug}$ 및 2세대 모델인 $\text{ST+sAug g2}$가 특히 열악한 환경에서 뛰어난 성능 향상을 보였다.

- **공격적 증강의 위험성**: 단순 지도 학습에 SpecAugment를 추가한 $\text{MP+sAug}$ 모델은 오히려 성능이 크게 저하되었다(예: Far-field Clean에서 FR $1.83\% \rightarrow 6.28\%$). 이는 하드 레이블 기반 학습에서 과도한 증강이 잘못된 레이블 정보를 제공하기 때문이다.
- **제안 방법의 효과**: $\text{ST+sAug g2}$ 모델은 Far-field Clean 환경에서 FR을 $1.83\%$에서 $0.78\%$로 낮추었으며, 특히 Query Logs 환경에서는 $8.21\%$에서 $3.12\%$로 약 $60\%$의 상대적 성능 향상을 기록하였다.
- **Noisy Student 방식과의 비교**: 교사에게는 깨끗한 데이터를 주고 학생에게만 증강된 데이터를 준 $\text{ST+sAug NS}$ 모델은 제안 방법보다 성능이 낮았다. 이는 교사가 증강된 데이터의 특성을 반영하지 못한 채 잘못된 소프트 레이블을 생성했기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 KWS와 같은 이진 분류 문제에서 데이터 증강을 적용할 때 발생하는 '레이블 불일치' 문제를 소프트 레이블과 Student-Teacher 구조로 해결할 수 있음을 입증하였다. 특히, 교사 모델에게도 동일한 노이즈를 부여하여 "데이터가 훼손되었음"을 인지하게 하고, 이를 통해 학생 모델이 유연하게 학습하도록 유도한 점이 핵심적인 통찰이다.

강점으로는 대규모 미분류 데이터를 효과적으로 활용하여 Far-field나 Accented voice와 같은 까다로운 조건에서의 성능을 크게 끌어올렸다는 점이다. 다만, 이 방법론이 성공하기 위해서는 1단계에서 기초가 되는 교사 모델($T_0$)이 어느 정도의 성능을 갖추고 있어야 한다는 전제가 필요하다. 또한, $\text{ST+sAug g2}$와 같이 반복적인 학습(Iteration)이 성능 향상에 기여한다는 점을 보여주었으나, 이에 따른 계산 비용 증가에 대한 논의는 명시되지 않았다.

## 📌 TL;DR

본 논문은 KWS 성능 향상을 위해 대규모 미분류 데이터와 공격적인 SpecAugment를 활용하는 **Noisy Student-Teacher** 학습법을 제안한다. 핵심은 교사와 학생 모두에게 동일한 증강 데이터를 입력하여, 데이터 훼손 정도가 반영된 소프트 레이블을 통해 학습하는 것이다. 이를 통해 특히 원거리(Far-field) 및 실제 쿼리 로그(QLog)와 같은 어려운 환경에서 기존 지도 학습 대비 최대 $60\%$의 성능 향상을 달성하였으며, 이는 향후 강건한 온디바이스 KWS 시스템 구축에 중요한 기여를 할 것으로 보인다.