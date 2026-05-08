# DomainSiam: Domain-Aware Siamese Network for Visual Object Tracking

Mohamed H. Abdelpakey and Mohamed S. Shehata (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 기존의 Siamese tracker들이 이미지 분류(image classification) 작업으로 사전 학습된 네트워크의 시맨틱(semantic) 및 객체성(objectness) 정보를 충분히 활용하지 못한다는 점이다.

일반적으로 이미지 분류 네트워크는 클래스 간의 차이를 극대화하고 클래스 내의 변화에는 둔감하도록 학습된다. 그러나 객체 추적(object tracking) 작업에서는 특정 클래스에 종속되지 않는 class-agnostic한 특성을 가지면서도, 동시에 대상 객체의 시맨틱한 정보를 정밀하게 활용해야 한다. 또한, 사전 학습된 Siamese 아키텍처는 카테고리 레이블에 의해 일부 채널만 활성화되는 sparsity(희소성) 문제가 발생하며, 이는 불필요한 계산량을 증가시키고 오버피팅(overfitting)을 유발하는 원인이 된다.

따라서 본 논문의 목표는 사전 학습된 네트워크의 시맨틱 정보를 최대한 활용하면서도, 클래스에 구애받지 않는 Domain-Aware 특성을 추출하여 추적 성능과 일반화 능력을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 **Ridge Regression Network**를 도입하여 가장 변별력 있는 컨볼루션 필터를 선택하고, 이를 통해 Domain-Aware한 특성을 추출하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Domain-Aware 아키텍처 제안**: 사전 학습된 네트워크에서 시맨틱 및 객체성 정보를 캡처하여 외형 변화에 강건하고 계산 효율적인 특성 공간을 생성하는 구조를 제안한다.
2. **미분 가능한 가중-동적 손실 함수(Differentiable Weighted-Dynamic Loss Function) 개발**: Ridge regression 네트워크를 학습시키기 위해 하이퍼파라미터에 따라 특성이 변하는 동적 손실 함수를 제안한다. 이는 학습 데이터의 불균형 문제를 해결하고 수렴 속도를 높인다.
3. **도메인 일반화 능력 향상**: 제안된 구조를 통해 ImageNet과 같은 일반 분류 데이터셋에서 학습된 지식을 VOT와 같은 추적 데이터셋으로 효율적으로 전이(generalization)시킬 수 있음을 보여준다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구를 검토한다.

- **Siamese-based Trackers**: SiamFC, SiamRPN, DensSiam 등은 타겟 브랜치와 서치 브랜치의 유사도 함수를 학습하여 객체를 추적한다. 특히 DensSiam은 Dense-connection과 Self-attention을 통해 깊은 네트워크에서도 그래디언트 소실 문제 없이 비지역적(non-local) 특성을 캡처한다. 그러나 이러한 트래커들은 대부분 사전 학습된 네트워크의 시맨틱 정보를 정적으로만 사용하며, 채널별 중요도를 고려하지 않는 한계가 있다.
- **Gradient-based Localization Guidance**: Grad-CAM과 같은 연구들은 특정 클래스에 대해 어떤 채널이 활성화되는지를 분석하여 객체의 위치를 파악한다. 본 논문은 이러한 직관을 확장하여, 추적 대상에 최적화된 도메인 채널을 동적으로 선택하는 방식을 채택한다.

## 🛠️ Methodology

### 전체 시스템 구조

DomainSiam의 전체 파이프라인은 크게 세 부분으로 구성된다.

1. **입력 블록**: 타겟 이미지와 서치 이미지를 입력으로 받는다.
2. **DensSiam 네트워크**: 특징 추출기(feature extractor)로 사용되며, 타겟 브랜치 끝에 Self-Attention 모듈이 배치되어 있다.
3. **Ridge Regression Network**: 추출된 특징 맵에서 가장 중요한 채널을 식별하여 Domain-Aware 특성을 생성한다. 최종적으로 이 특성들이 상관관계 층(correlation layer)을 통해 응답 맵(response map)을 생성한다.

### Ridge Regression Network 및 Domain-Aware 특성

사전 학습된 네트워크의 채널 중 일부만 활성화되는 sparsity 문제를 해결하기 위해, Ridge regression을 통해 각 채널의 중요도를 계산한다. 네트워크는 입력 이미지 패치의 샘플들을 소프트 레이블(soft label)로 회귀시킨다. 이때 목적 함수는 다음과 같다.

$$\arg \min_{w} \|W * X_{i,j} - Y(i,j)\|^2 + \lambda \|W\|^2$$

여기서 $W$는 회귀 네트워크의 가중치, $X$는 입력 특성, $Y$는 소프트 레이블이며, $\lambda$는 정규화 파라미터이다. 소프트 레이블 $Y(i,j)$는 타겟 중심을 기준으로 하는 가우시안 분포(Gaussian distribution)를 사용한다.

$$Y(i,j) = e^{-\frac{i^2 + j^2}{2\sigma^2}}$$

최종적으로 Domain-Aware 특성 $\delta_i$는 손실 함수 $L$에 대한 입력 특성 채널 $F_i$의 그래디언트에 전역 평균 풀링(Global Average Pooling, GAP)을 적용하여 계산한다.

$$\delta_i = \text{GAP}\left(\frac{\partial L}{\partial F_i}\right)$$

### 미분 가능한 가중-동적 손실 함수

본 논문은 Ridge regression 최적화를 위해 다음과 같은 가중-동적 손실 함수 $L(x, \alpha)$를 제안한다.

$$L(x, \alpha) = \frac{|\alpha - 2|}{\alpha} e^{ay} \left( \left( \frac{x^2}{|\alpha - 2|} + 1 \right)^{\alpha/2} - 1 \right)$$

여기서 $a \in [0, 1]$는 하이퍼파라미터이며, $\alpha$는 손실 함수의 강건성(robustness)을 제어하는 파라미터이다. 이 함수는 $\alpha$ 값에 따라 다음과 같이 변환된다.

- $\alpha = 2$일 때: $L^2$ 손실 함수로 수렴한다.
- $\alpha = 1$일 때: $L^1$ 손실 함수와 유사한 형태로 변한다.
- $\alpha = 0$일 때: Lorentzian 손실 함수로 수렴한다.

이 손실 함수는 지수 항($e^{ay}$)을 통해 배경(easy samples)보다 전경(hard samples)에 더 높은 가중치를 부여함으로써 데이터 불균형 문제를 해결하며, $\alpha$에 대해 단조 증가(monotonic)하는 특성을 가져 비볼록(non-convex) 최적화 환경에서도 유용하다.

## 📊 Results

### 실험 설정

- **데이터셋**: 검증 셋으로 OTB2013, OTB2015를 사용하고, 테스트 셋으로 VOT2017, VOT2018, LaSOT, TrackingNet, GOT10k를 사용하였다.
- **구현 세부사항**: ILSVRC15로 사전 학습된 DensSiam 네트워크를 기반으로 하며, Ridge regression 네트워크는 70 에포크 동안 별도로 학습되었다. 가장 높은 점수를 가진 100개의 채널을 Domain-Aware 특성으로 선택하였다.
- **성능 지표**: 정확도(Accuracy), 기대 평균 겹침(EAO), 강건성(Robustness), 처리 속도(FPS) 등을 측정하였다.

### 정량적 결과

- **VOT2017/2018**: DomainSiam은 VOT2017에서 Accuracy(0.562)와 EAO(0.374) 면에서 SOTA 성능을 달성하였으며, VOT2018에서도 Accuracy(0.593), EAO(0.396), Robustness(0.221)에서 경쟁 모델들을 압도하였다.
- **TrackingNet**: 정밀도(Precision) 0.585, 성공률(Success) 0.635를 기록하며 MDNet 등 기존 모델보다 우수한 성능을 보였다.
- **LaSOT**: 성공률(Success) 43.6%를 기록하여 DaSiam(41.5%)보다 높은 성능을 보였다.
- **GOT10k**: 평균 겹침(AO) 0.414를 기록하며 CFNet(0.374) 대비 약 4% 향상된 결과를 보였다.
- **속도**: 53 FPS의 속도로 동작하여 실시간 추적이 가능하다.

## 🧠 Insights & Discussion

### 강점 및 성과

DomainSiam은 Ridge regression 네트워크를 통해 사전 학습된 모델의 방대한 채널 중 추적 대상에 가장 유용한 채널만을 동적으로 선택함으로써, 계산 효율성을 높이는 동시에 변별력을 강화하였다. 특히 제안된 동적 손실 함수는 $L^2$나 Shrinkage loss보다 훨씬 빠른 수렴 속도를 보였으며, 이는 학습 데이터의 불균형을 효과적으로 처리했기 때문으로 분석된다. 또한, 특정 도메인(ImageNet)에서 학습된 가중치를 다른 도메인(VOT 등)으로 전이할 때 발생하는 성능 저하 문제를 Domain-Aware 특성 추출을 통해 완화하였다.

### 한계 및 논의

논문에서는 Ridge regression 네트워크를 Siamese 네트워크와 별도로 학습시킨다고 명시하고 있다. 이는 전체 시스템을 end-to-end로 학습시키는 방식에 비해 최적화 관점에서 한계가 있을 수 있다. 또한, 100개의 채널을 선택하는 기준이 경험적인 설정일 가능성이 크므로, 대상 객체의 복잡도에 따라 최적의 채널 수가 달라질 수 있다는 점이 미해결 질문으로 남는다.

## 📌 TL;DR

본 논문은 Siamese tracker의 고질적인 문제인 사전 학습 네트워크의 시맨틱 정보 활용 부족과 채널 희소성(sparsity) 문제를 해결하기 위해 **DomainSiam**을 제안한다. Ridge regression 네트워크와 새로운 동적 손실 함수를 도입하여 객체 추적에 최적화된 **Domain-Aware 특성**을 추출하며, 이를 통해 VOT, LaSOT, GOT10k 등 주요 벤치마크에서 SOTA 성능을 달성함과 동시에 53 FPS의 실시간성을 확보하였다. 이 연구는 사전 학습된 대규모 모델의 지식을 특정 하위 작업(downstream task)에 맞춰 효율적으로 필터링하는 방법론을 제시했다는 점에서 향후 객체 검출 및 세그멘테이션 연구에도 기여할 가능성이 크다.
