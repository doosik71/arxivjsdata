# Instance Consistency Regularization for Semi-Supervised 3D Instance Segmentation

Yizheng Wu, Zhiyu Pan, Kewei Wang, Xingyi Li, Jiahao Cui, Liwen Xiao, Guosheng Lin, Zhiguo Cao (2024)

## 🧩 Problem to Solve

본 논문은 3D 인스턴스 세그멘테이션(3D Instance Segmentation)을 위한 준지도 학습(Semi-supervised Learning) 환경에서의 성능 향상을 목표로 한다. 3D 데이터셋에 포인트 수준의 시맨틱 및 인스턴스 라벨을 부여하는 작업은 매우 많은 시간과 비용이 소모되므로, 라벨이 없는(unlabeled) 데이터를 효율적으로 활용하는 것이 매우 중요하다.

기존의 준지도 학습 접근 방식들은 주로 셀프 트레이닝(Self-training) 프레임워크를 사용하여 의사 라벨(Pseudo labels)을 생성하고, 이를 통해 일관성 규제(Consistency Regularization)를 수행한다. 그러나 기존 방식들은 시맨틱 의사 라벨과 인스턴스 의사 라벨을 동시에 사용하는 공동 학습(Joint learning) 방식을 취하는데, 여기서 **시맨틱 모호성(Semantic Ambiguity)** 문제가 발생한다. 구체적으로, 클래스 분포의 불균형과 유사한 카테고리(예: 의자와 소파) 간의 혼동으로 인해 시맨틱 의사 라벨에 많은 노이즈가 포함되며, 이는 결국 셀프 트레이닝 과정에서 모델의 붕괴(Collapse)를 초래한다.

따라서 본 논문은 3D 인스턴스가 서로 겹치지 않고 공간적으로 분리 가능하다는 특성에 주목하여, 노이즈가 많은 시맨틱 정보에 의존하지 않고 오직 **인스턴스 일관성 규제(Instance Consistency Regularization)**만을 활용하여 준지도 학습 성능을 높이는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시맨틱 의사 라벨을 과감히 배제하고, 더 신뢰할 수 있는 인스턴스 의사 라벨만을 사용하여 모델을 학습시키는 것이다. 이를 위해 다음과 같은 기여를 한다.

1.  **시맨틱 모호성 식별**: 준지도 3D 인스턴스 세그멘테이션에서 의사 라벨의 노이즈 근원이 시맨틱 모호성임을 밝혀냈다.
2.  **DKNet 설계**: 시맨틱 세그멘테이션 브랜치에 의존하지 않고, 판별적인 인스턴스 커널(Discriminative instance kernels)을 통해 인스턴스를 구분하는 병렬 구조의 3D 인스턴스 세그멘테이션 모델인 DKNet을 제안하였다.
3.  **InsTeacher3D 프레임워크**: 고품질의 인스턴스 의사 라벨을 생성하고 활용하는 셀프 트레이닝 네트워크인 InsTeacher3D를 제안하였다.
4.  **실험적 검증**: ScanNetV2, S3DIS, STPLS3D 등 대규모 데이터셋에서 기존 SOTA(State-of-the-art) 준지도 학습 방식보다 뛰어난 성능을 입증하였다.

## 📎 Related Works

### 3D Instance Segmentation
기존의 3D 인스턴스 세그멘테이션은 크게 세 가지 방향으로 발전해 왔다.
- **Proposal-based**: 객체 제안(Proposal)을 먼저 생성한 후 내부 포인트를 분리하는 방식이다.
- **Proposal-free/Clustering-based**: 임베딩 유사도를 기반으로 포인트를 그룹화하거나, 시맨틱 예측에 직렬적으로 의존하여 휴리스틱 클러스터링을 수행하는 방식이다. 하지만 이러한 직렬적 의존성은 시맨틱 예측의 오류가 인스턴스 분리 오류로 이어지는 한계가 있다.
- **Transformer-based**: 쿼리 인코더를 통해 인스턴스를 쿼리 형태로 인코딩하는 방식이다. 그러나 대규모 데이터셋에 적합하도록 설계되어, 데이터 효율적인(Data-efficient) 설정에서는 성능이 저하되는 경향이 있다.

### Data-Efficient 3D Scene Understanding
제한된 라벨을 사용하는 설정은 크게 Limited Annotation(LA)과 Limited Reconstruction(LR)으로 나뉜다. 본 논문은 일부 씬(Scene)만 완전히 라벨링된 LR 설정에 집중한다. TWIST와 같은 기존 준지도 학습 모델들이 시맨틱과 인스턴스 라벨을 모두 활용하려 했으나, 앞서 언급한 시맨틱 모호성으로 인해 한계가 있었다.

## 🛠️ Methodology

### 전체 시스템 구조
InsTeacher3D는 **Mean Teacher** 프레임워크를 기반으로 한다. 학생 모델($\Phi_s$)은 강한 증강(Strong augmentation)이 적용된 데이터를 학습하고, 교사 모델($\Phi_t$)은 약한 증강(Weak augmentation)이 적용된 데이터로부터 고품질의 의사 라벨을 생성하여 학생 모델을 가이드한다. 교사 모델의 가중치는 학생 모델의 가중치를 지수 이동 평균(EMA, Exponential Moving Average)하여 업데이트한다.

$$\Phi_{t}^{\tau+1} = \alpha \cdot \Phi_{t}^{\tau} + (1-\alpha) \cdot \Phi_{s}^{\tau+1}$$

### DKNet (Base Model)
DKNet은 시맨틱 예측과 병렬로 동작하여 인스턴스 마스크를 생성하는 모델이다. 다음의 3단계 파이프라인을 가진다.
1.  **Instance Localization**: 센트로이드 히트맵($H$)과 오프셋($O$)을 예측하여 인스턴스의 중심점 후보를 찾는다.
2.  **Instance Representation**: 후보점 주변의 정보를 수집하여 인스턴스 커널을 구축한다. 이때 Affinity Matrix $A$를 통해 중복된 후보를 병합한다.
3.  **Instance Reconstruction**: 구축된 다이내믹 커널(Dynamic kernels)을 이용해 전체 씬을 스캔하여 소프트 마스크 $R$을 재구성한다.

### Dynamic Mask Generation (DMG)
교사 모델이 생성한 소프트 마스크 $R$은 노이즈가 많으므로, 이를 고품질의 하드 의사 라벨 $\hat{M}$으로 변환하는 DMG 모듈을 사용한다.
- **Intra-Instance Self-Enhancement**: Otsu 알고리즘을 이용해 인스턴스별 동적 임계값($T_i$)을 설정함으로써, 신호가 약한 인스턴스가 무시되는 것을 방지한다.
- **Inter-Instance Self-Enhancement**: 소프트 마스크와 하드 마스크 간의 일관성을 측정하는 Purity Score $S_{purity}$를 계산하여, 중복되거나 노이즈가 심한 마스크를 제거한다.
  $$S_{purity} = \frac{\sum_{n=1}^{N} R_{n,i} \cdot \mathbb{1}(\hat{M}_n = i)}{\sum_{n=1}^{N} R_{n,i} \cdot \mathbb{1}(R_{n,i} > 0.5)}$$
- **Superpoint Refinement**: 공간적 인접성을 고려하여 슈퍼포인트 단위로 라벨을 보정함으로써 마스크의 완결성을 높인다.

### 학습 목표 및 손실 함수
모델은 라벨링된 데이터에 대한 지도 학습 손실($\mathcal{L}_l$)과 라벨링되지 않은 데이터에 대한 일관성 손실($\mathcal{L}_u$)을 동시에 최소화한다.

$$\mathcal{L} = \mathcal{L}_{l}^{sem} + \mathcal{L}_{l}^{ins} + \mathcal{L}_{u}^{ins}$$

여기서 인스턴스 관련 손실 $\mathcal{L}_{ins}$는 다음과 같이 구성된다.
$$\mathcal{L}_{ins} = \mathcal{L}_{loc} + \mathcal{L}_{rep} + \mathcal{L}_{rec}$$
- $\mathcal{L}_{loc}$: 센트로이드 오프셋과 히트맵에 대한 손실이다.
- $\mathcal{L}_{rep}$: 인스턴스 후보 간의 유사도를 학습하는 Binary Cross Entropy(BCE) 손실이다.
- $\mathcal{L}_{rec}$: 재구성된 마스크와 정답 마스크 간의 BCE 및 Dice 손실이다.

## 📊 Results

### 실험 설정
- **데이터셋**: ScanNetV2, S3DIS, STPLS3D.
- **설정**: 라벨링 비율을 1%, 5%, 10%, 20%로 설정한 Limited Reconstruction(LR) 환경.
- **지표**: mAP, $\text{AP}_{50}$, $\text{AP}_{25}$.

### 주요 결과
- **성능 향상**: InsTeacher3D는 모든 데이터셋에서 기존 SOTA 준지도 학습 모델보다 우수한 성능을 보였다. 특히 ScanNetV2의 20% 라벨 설정에서 mAP 42.7%를 달성하여, TWIST(32.8%) 대비 9.9%p 높은 성능을 기록했다.
- **데이터 효율성**: 20%의 라벨만 사용한 InsTeacher3D가 일부 완전 지도 학습(Fully-supervised) 모델(예: PointGroup, GSPN)보다 높은 mAP를 기록하며 뛰어난 데이터 효율성을 입증했다.
- **야외 데이터셋 확장성**: STPLS3D 데이터셋에서도 20% 라벨 사용 시 완전 지도 학습 기반 DKNet 성능의 93.4%까지 도달하는 강건함을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **인스턴스 일관성의 우위**: 본 논문은 시맨틱 의사 라벨을 사용하는 것보다 인스턴스 의사 라벨만을 사용하는 것이 훨씬 안정적인 학습을 가능하게 함을 입증했다. 이는 3D 데이터에서 인스턴스의 기하학적 분리성이 시맨틱 카테고리의 구별보다 더 명확하기 때문이다.
- **병렬 구조의 중요성**: DKNet과 같은 병렬 구조는 시맨틱 예측의 오류가 인스턴스 분리로 전이되는 것을 차단하여, 준지도 학습 환경에서 훨씬 더 깨끗한 의사 라벨을 생성할 수 있게 한다.
- **센트로이드 집중 전략**: 센트로이드 영역의 특징을 활용하는 것이 경계 영역의 노이즈를 효과적으로 배제하여 인스턴스 커널의 판별력을 높인다는 점을 확인했다.

### 한계 및 논의사항
- **저라벨 비율에서의 취약성**: 1%와 같이 극단적으로 적은 라벨 설정에서는 DKNet의 커널 기반 패러다임이 충분한 인스턴스 매칭 쌍을 찾지 못해 성능이 저하되는 경향이 있다. 이를 해결하기 위해 CSC와 같은 대조 학습(Contrastive Learning) 기법을 결합하여 보완할 수 있음을 제시하였다.
- **가정**: 본 모델은 인스턴스가 공간적으로 잘 분리되어 있다는 가정을 전제로 한다. 만약 객체들이 매우 밀집되어 있거나 겹쳐 있는 특수한 환경에서는 인스턴스 일관성 규제의 효과가 감소할 가능성이 있다.

## 📌 TL;DR

본 논문은 3D 인스턴스 세그멘테이션의 준지도 학습에서 발생하는 **시맨틱 모호성(Semantic Ambiguity)** 문제를 해결하기 위해, 시맨틱 라벨을 배제하고 **인스턴스 일관성 규제(Instance Consistency Regularization)**만을 활용하는 **InsTeacher3D**를 제안한다. 병렬 구조의 **DKNet**과 고품질 의사 라벨을 생성하는 **DMG 모듈**을 통해, 적은 양의 라벨만으로도 SOTA 수준의 인스턴스 분리 성능을 달성하였으며, 이는 3D 씬 이해에서 기하학적 인스턴스 정보가 시맨틱 정보보다 더 신뢰할 수 있는 가이드가 됨을 시사한다.