# Scale Decoupled Distillation

Shicai Wei, Chunbo Luo, Yang Luo (2024)

## 🧩 Problem to Solve

본 논문은 Logit 기반 지식 증류(Logit Knowledge Distillation)에서 발생하는 성능 저하 문제를 해결하고자 한다. 기존의 Logit 기반 방식은 주로 네트워크의 최종 출력인 Global Logit을 사용하는데, 이 Global Logit은 이미지 전체의 정보를 통합하고 있어 여러 클래스의 세부적인 시맨틱 지식이 서로 얽혀 있는(coupled) 상태이다.

특히, 서로 다른 클래스가 동일한 상위 클래스(super-class)에 속해 전역적 특징이 유사하거나, 하나의 이미지 내에 여러 클래스의 정보가 동시에 존재하는 경우, Global Logit은 모호한(ambiguous) 지식을 전달하게 된다. 이러한 모호함은 학생(Student) 모델의 학습을 방해하고 오분류를 유도하며, 결과적으로 Feature 기반 지식 증류에 비해 성능이 떨어지는 원인이 된다. 따라서 본 연구의 목표는 Global Logit의 모호성을 제거하고, 정밀하고 명확한 지식을 전달할 수 있는 Scale Decoupled Distillation (SDD) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Global Logit 출력을 다양한 스케일의 Local Logit 출력으로 분리(Decouple)하여 전달하는 것이다.

1. **Scale Decoupling**: 이미지 전체의 Global Logit을 여러 지역적 영역의 Local Logit으로 분리함으로써, 얽혀 있던 세부 시맨틱 지식을 추출하고 모호성을 줄인다.
2. **Knowledge Division**: 분리된 Local Logit 지식을 Global Logit과 일치하는 'Consistent' 지식과 일치하지 않는 'Complementary' 지식으로 구분한다.
3. **Ambiguity Weighting**: Complementary 지식에 더 높은 가중치를 부여하여, 학생 모델이 모호한 샘플에 더 집중하게 함으로써 판별 능력을 향상시킨다.
4. **Structural Efficiency**: 추가적인 분류기나 복잡한 아키텍처 변경 없이 기존 분류기를 그대로 활용하여 Multi-scale Logit을 생성하므로 연산 효율성이 높다.

## 📎 Related Works

### Feature-based Distillation

FitNets를 시작으로 AT, RKD, SP 등 중간 특징 맵(Intermediate feature map)이나 어텐션 맵을 직접 일치시키는 방식이 제안되었다. 이러한 방법들은 높은 성능을 보이지만, 교사(Teacher)와 학생 모델의 아키텍처가 서로 다른(Heterogeneous) 경우, 특히 레이어 수가 다를 때 성능이 저하되는 한계가 있다.

### Logit-based Distillation

Hinton이 제안한 기본 KD를 시작으로, 동적 온도 조절(FN), 자기 지도 학습 기반의 분류기 추가(SSKD), 지식 분리(DKD, NKD) 등이 제안되었다. DKD는 Logit을 Target 클래스와 Non-target 클래스로 분리하여 전이하지만, 여전히 이미지 전체의 Global Logit만을 사용하기 때문에 지역적인 세부 정보에 포함된 모호성 문제를 해결하지 못한다는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인

SDD는 기존의 Global Average Pooling 기반 KD를 확장하여, **Multi-scale Pooling**과 **Information Weighting**이라는 두 가지 핵심 단계로 구성된다.

### 1. Multi-scale Pooling 및 Local Logit 생성

먼저 교사와 학생 네트워크의 특징 추출기($f_{Net}$)와 프로젝션 행렬($W_{Net}$)을 정의한다. 입력 이미지 $x$에 대해 최종 특징 맵 $f_{Net}(x)$의 각 위치 $(j, k)$에서의 벡터를 $f_{Net}(j, k)$라고 할 때, 이를 $W_{Net}$에 곱해 Logit 맵 $L_{Net}$을 얻는다.

SDD는 이 $L_{Net}$을 다양한 스케일 $m \in M$으로 나누어 평균 풀링을 수행한다. 특정 스케일 $m$의 $n$번째 셀 $C(m, n)$에 대한 Local Logit $\pi_{Net}(m, n)$은 다음과 같이 계산된다.

$$\pi^T(m, n) = \sum_{j, k \in C(m, n)} \frac{1}{m^2} L^T(j, k)$$
$$\pi^S(m, n) = \sum_{j, k \in C(m, n)} \frac{1}{m^2} L^S(j, k)$$

여기서 $L^T$와 $L^S$는 각각 교사와 학생의 Logit 맵이다. 각 지역 영역 $Z(m, n)$에 대한 증류 손실 $D(m, n)$은 기존의 Logit 기반 손실 함수 $L_D$(예: KL Divergence)를 사용하여 정의한다.

$$D(m, n) = L_D(\sigma(\pi^T(m, n)), \sigma(\pi^S(m, n)))$$

### 2. Information Weighting (Consistent vs Complementary)

분리된 Local Logit들을 Global Logit의 예측 클래스와 비교하여 두 그룹으로 나눈다.

- **Consistent terms ($D_{con}$)**: Global Logit과 동일한 클래스를 가진 Local Logit들로, 해당 클래스의 다중 스케일 지식을 전달한다.
- **Complementary terms ($D_{com}$)**: Global Logit과 다른 클래스를 가진 Local Logit들로, 샘플의 모호성(Ambiguity)을 보존한다.

최종 SDD 손실 함수 $L_{SDD}$는 다음과 같이 정의된다.

$$L_{SDD} = D_{con} + \beta D_{com}$$

여기서 $\beta$는 Complementary 지식에 부여하는 가중치이며, 이를 통해 학생 모델이 모호한 샘플을 더 잘 학습하도록 유도한다.

### 3. 전체 학습 목표

최종 손실 함수는 정답 레이블에 대한 Cross Entropy 손실($L_{CE}$)과 SDD 손실의 가중 합으로 구성된다.

$$L = L_{CE} + \alpha L_{SDD}$$

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-100, ImageNet, 그리고 세밀한 분류가 필요한 CUB200(조류 200종)을 사용하였다.
- **비교 대상**: FitNet, SP, CRD, SemCKD, ReviewKD, MGD 및 Logit 기반의 KD, DKD, NKD를 비교 대상으로 설정하였다.
- **구현**: $\beta=2.0$으로 설정하였으며, 스케일 셋 $M$은 아키텍처 유사도에 따라 $\{1, 2, 4\}$ 또는 $\{1, 2\}$로 설정하였다.

### 주요 결과

1. **범용적 성능 향상**: 다양한 교사-학생 쌍(Heterogeneous 및 Homogeneous)에서 SDD를 적용했을 때, 기본 KD, DKD, NKD의 성능이 일관되게 향상되었다. 특히 CIFAR-100에서 SD-DKD는 기존의 SOTA Feature 기반 방식인 ReviewKD와 MGD보다 높은 성능을 기록하였다.
2. **Fine-grained Classification에서의 강점**: CUB200 데이터셋에서 SDD는 $1.06\% \sim 6.41\%$라는 매우 큰 성능 향상을 보였다. 이는 세밀한 분류 작업일수록 전역 정보보다 지역적인 세부 시맨틱 정보가 중요하기 때문으로 분석된다.
3. **효율성**: SDD는 추가적인 분류기를 생성하지 않고 기존 분류기를 재사용하여 Local Logit을 계산하므로, 학습 시간이 기본 KD와 동일하며 Feature 기반 방식보다 빠르다.

### Ablation Study

- **지식 분리의 효과**: Consistent 지식과 Complementary 지식을 모두 사용했을 때(Fusion) 가장 높은 성능이 나타났으며, 두 요소 모두 단독으로도 기본 KD보다 성능을 높였다.
- **스케일 설정**: Heterogeneous 구조의 경우 더 세밀한 지식이 필요하여 $M=\{1, 2, 4\}$에서 최적의 성능을 보였으나, Homogeneous 구조에서는 정보 중복으로 인해 $M=\{1, 2\}$가 더 유리하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 Global Logit이 가진 '정보의 얽힘' 문제를 공간적 스케일 분리(Scale Decoupling)라는 간단한 방법으로 해결하였다. 특히 Complementary 지식을 통해 교사 모델이 가진 모호성을 학생 모델이 적절히 학습하게 함으로써, 단순한 모방을 넘어 판별 능력을 강화한 점이 돋보인다. t-SNE 시각화 결과, SD-KD로 학습된 특징들이 KD보다 더 잘 분리되는 것을 통해 판별력 향상이 증명되었다.

### 한계 및 논의사항

논문에서는 스케일 셋 $M$을 하이퍼파라미터로 설정하여 최적의 값을 찾았으나, 데이터셋이나 모델 구조에 따라 최적의 스케일을 자동으로 결정하는 메커니즘에 대해서는 명시되지 않았다. 또한, $\beta$ 값에 따른 성능 변화를 실험하였으나, 이 값이 갖는 이론적 근거보다는 실험적 최적값($\beta=2.0$)에 의존하고 있다는 점이 한계로 보인다.

## 📌 TL;DR

본 논문은 Global Logit 기반 지식 증류가 가진 시맨틱 모호성 문제를 해결하기 위해, Logit 출력을 다중 스케일의 지역적(Local) 출력으로 분리하는 **Scale Decoupled Distillation (SDD)**을 제안한다. SDD는 지식을 Consistent와 Complementary 그룹으로 나누어 전이하며, 특히 모호한 샘플에 더 높은 가중치를 두어 학습시킨다. 실험 결과, 추가적인 연산 비용 없이 다양한 모델 구조에서 성능 향상을 이루었으며, 특히 세밀한 분류(Fine-grained classification) 작업에서 매우 탁월한 효과를 보였다. 이는 향후 정밀한 객체 인식 모델의 경량화 연구에 중요한 기여를 할 것으로 보인다.
