# Deep Level Set for Box-supervised Instance Segmentation in Aerial Images

Wentong Li, Yijie Chen, Wenyu Liu, Jianke Zhu (2021)

## 🧩 Problem to Solve

본 논문은 항공 이미지(Aerial Images)에서의 **Box-supervised Instance Segmentation (BSIS)** 문제를 해결하고자 한다. 일반적으로 인스턴스 분할(Instance Segmentation)은 픽셀 수준의 마스크 주석(Mask Annotation)을 필요로 하지만, 이는 비용과 시간이 매우 많이 소요된다. 이에 따라 바운딩 박스(Bounding Box)만으로 마스크를 예측하는 BSIS 연구가 진행되어 왔다.

항공 이미지 도메인은 일반적인 객체 데이터셋과 비교하여 다음과 같은 고유한 난제들이 존재한다.
1. **큰 클래스 내 변동성(Intra-class variance)** 및 **클래스 간 유사성(Inter-class similarity)**: 서로 다른 객체가 비슷하게 보이거나, 같은 클래스 내에서도 외형 차이가 크다.
2. **복잡한 배경 및 미세 객체(Tiny objects)**: 고해상도 위성 이미지 특성상 배경이 복잡하며 매우 작은 객체들이 많이 분포한다.

기존의 BSIS 방법론들은 주로 **Pairwise Affinity Modeling**(인접 픽셀 간의 유사도 모델링)에 의존하는데, 항공 이미지의 복잡한 배경과 객체 특성으로 인해 이러한 방식은 노이즈가 섞인 감독 신호(Noisy supervision)를 생성하며, 결과적으로 정밀도가 떨어지는 문제가 발생한다. 따라서 본 논문의 목표는 이러한 노이즈 간섭을 최소화하고 정확한 객체 경계를 복원할 수 있는 새로운 항공 이미지 인스턴스 분할 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 픽셀 쌍 유사도 모델링 대신, **Level Set Method**를 딥러닝 네트워크에 통합하여 객체 분할을 **곡선 진화(Curve Evolution)** 과정으로 처리하는 것이다.

주요 기여 사항은 다음과 같다.
1. **Deep Level Set 기반의 BSIS 접근법 제안**: 바운딩 박스 주석만을 사용하여 항공 이미지의 인스턴스 분할을 수행하는 최초의 딥 레벨 셋 방법론을 제시하였다.
2. **통합된 샘플 할당 프레임워크**: Detection 브랜치와 Segmentation 브랜치가 동일한 방식으로 잠재적 양성 샘플(Potential positive samples)을 선택하고, 분류, 박스 회귀, 마스크 예측을 공동으로 학습하는 구조를 설계하였다.
3. **성능 입증**: iSAID 및 Potsdam 데이터셋에서 실험을 통해 기존의 Box-supervised 방법론들을 상회하는 성능을 보였으며, 일부 카테고리에서는 Fully mask-supervised 방법론에 근접하는 결과를 달성하였다.

## 📎 Related Works

### Level Set-based Segmentation
Level Set 방법은 고차원 함수 $\phi$의 제로 레벨 셋(Zero level set)을 통해 암시적으로 곡선을 표현하고, 에너지 함수를 최소화하는 방향으로 곡선을 진화시켜 경계를 찾는 기법이다. 최근에는 이를 딥러닝에 결합하여 정밀한 분할을 시도하는 연구들이 있었으나, 대부분은 정답 마스크가 존재하는 Fully supervised 환경에서만 작동하였다.

### Box-Supervised Instance Segmentation
최근 BoxInst나 BBTP와 같은 방법들이 제안되었다. 이들은 주로 픽셀 간의 색상이나 공간적 유사성을 그래프로 모델링하여 마스크를 생성한다. 하지만 이러한 Pairwise affinity 방식은 "인접한 픽셀은 같은 라벨을 가질 확률이 높다"는 지나치게 단순한 가정에 기반하므로, 항공 이미지처럼 배경이 복잡한 경우 배경 노이즈가 마스크에 포함되는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조
제안된 모델은 ResNet-50과 FPN(Feature Pyramid Network)을 백본으로 사용하며, 두 개의 헤드 브랜치로 구성된다.
1. **Detection Branch**: 객체의 임의 방향성을 고려하여 **Oriented Bounding Boxes (OBB)** 표현법을 사용한다. RPN을 통해 회전된 RoI를 생성하고, **Snake module**을 통해 정점(Vertices)들을 반복적으로 정교화하여 최종 박스를 예측한다.
2. **Segmentation Branch**: FPN의 최상위 해상도 특징 맵($P_2$)과 상위 시맨틱 특징들을 결합하여 인스턴스별 마스크 맵을 생성한다.

### Deep Level Set Formulation
본 논문은 객체의 경계를 $\phi(x,y)=0$인 지점으로 정의하며, $\phi > 0$은 내부, $\phi < 0$은 외부로 간주한다. 네트워크는 에너지 함수를 최소화하도록 $\phi$를 학습하며, 이때 Heaviside 함수 대신 미분 가능한 **Sigmoid 함수** $\sigma(\phi)$를 특성 함수로 사용하여 학습의 안정성을 높였다.

학습 시, 바운딩 박스 $B$를 확장한 영역 $B^*$ 내에서 곡선 진화를 수행한다. 에너지 함수 $L$은 다음과 같이 정의된다.

$$L(C_1, C_2, \phi, \rho_{cls}, B^*) = \alpha_1 \sum_n \rho_{cls} \int_{\Omega \in B^*} |u^*_0(x,y) - a^*_1|^2 \sigma(\phi(x,y)) dxdy + \alpha_2 \sum_n \rho_{cls} \int_{\Omega \in B^*} |u^*_0(x,y) - a^*_2|^2 (1 - \sigma(\phi(x,y))) dxdy + \lambda \text{Length}(\phi) + \mu \text{Area}(\phi)$$

여기서 각 항의 의미는 다음과 같다.
- **Fitting Terms (첫 번째, 두 번째 항)**: 내부($a^*_1$)와 외부($a^*_2$)의 평균값과 실제 이미지 값 $u^*_0$ 사이의 차이를 최소화하여 마스크가 객체 영역과 배경 영역으로 균일하게 나뉘도록 강제한다.
- **Length term ($\text{Length}(\phi)$)**: 경계선의 길이를 제어하여 분할 결과가 매끄럽게(Smooth) 유지되도록 하는 정규화 항이다.
- **Area term ($\text{Area}(\phi)$)**: 내부 영역의 넓이를 제어하는 정규화 항이다.
- **$\rho_{cls}$**: 클래스별 특성이 다르므로, 클래스마다 다른 진화 정도를 부여하는 가중치 파라미터이다.

### Box and Background Constraints
Level set의 수렴 속도를 높이고 노이즈를 방지하기 위해 두 가지 제약 조건을 추가하였다.
1. **Box Constraint ($L_{box\_cons}$)**: 예측된 마스크가 정답 박스 영역 내에 존재해야 한다는 제약이다. 마스크 예측값 $m_p$와 정답 박스 $b_f$를 $x$축과 $y$축으로 각각 투영(Projection)하여 그 차이를 계산한다.
2. **Background Constraint ($L_{back\_cons}$)**: 정답 박스 외부의 모든 영역은 배경($m_b = 1 - m_p$)이어야 한다는 제약이다.

두 제약 조건은 **Dice Loss**를 통해 구현되며, 최종 세그멘테이션 손실 함수는 다음과 같다.
$$L_{seg} = L_{levelset} + L_{cons}$$

## 📊 Results

### 실험 설정
- **데이터셋**: iSAID(15개 클래스), Potsdam(Car 클래스 중심)
- **지표**: Mask Average Precision (AP), $AP_{50}$, $AP_{75}$
- **구현**: ResNet-50 백본, SGD 옵티마이저, $800 \times 800$ 패치 크기 사용.

### 정량적 결과
1. **iSAID 데이터셋**:
   - 제안 방법은 Box-supervised 방법론인 **BoxInst 대비 AP 5.3% 상승**, **BBTP 대비 AP 3.4% 상승**하였다.
   - 특히 Basketball Court, Ground Track Field, Bridge, Large Vehicle 클래스에서는 Fully supervised 방법론과 경쟁 가능한 수준의 성능을 보였다.
2. **Potsdam 데이터셋**:
   - 제안 방법은 **47.3% AP**를 기록하며, BBTP(29.7%)와 BoxInst(38.4%)를 크게 상회하였다.

### Ablation Study 결과
- **손실 함수 영향**: $L_{cons}$만 사용했을 때(17.8% AP)보다 $L_{levelset}$을 추가했을 때(25.3% AP), 그리고 두 가지를 모두 사용했을 때(34.8% AP) 성능이 가장 높았다.
- **확장 영역 $B^*$**: 박스 영역을 2.0배 확장했을 때 가장 좋은 성능(36.1% AP)을 보였으며, 너무 넓으면($2.5\times$) 노이즈 유입으로 성능이 소폭 하락하였다.
- **$\rho_{cls}$ 설정**: 고정된 값보다 **Self-adaptive**하게 학습하는 방식이 가장 높은 성능(36.5% AP)을 기록하였다.
- **샘플 할당**: Max-IoU 방식보다 **ATSS** 방식을 사용했을 때 더 높은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점
본 논문은 단순한 픽셀 쌍의 유사도를 계산하는 기존 BSIS 방식의 한계를 정확히 짚어냈으며, 이를 해결하기 위해 고전적인 Level Set 이론을 딥러닝에 성공적으로 통합하였다. 특히, 곡선 진화(Curve Evolution)라는 접근 방식을 통해 항공 이미지의 복잡한 배경 속에서도 객체의 경계를 더 정밀하게 찾아낼 수 있음을 입증하였다. 또한, 투영 기반의 박스 제약 조건을 도입하여 weakly-supervised 학습의 불안정성을 효과적으로 보완하였다.

### 한계 및 논의사항
- **일반화 가능성**: 본 연구는 항공 이미지라는 특수한 도메인에서 수행되었다. 저자들도 결론에서 언급했듯이, COCO와 같은 일반적인 자연 이미지 데이터셋에서도 동일한 효과가 나타날지는 추가 검증이 필요하다.
- **계산 복잡도**: Level set의 반복적인 진화 과정이 추론 및 학습 단계에서 계산 비용을 증가시킬 가능성이 있으나, 이에 대한 구체적인 시간 복잡도 분석은 논문에 명시되지 않았다.
- **하이퍼파라미터 민감도**: $\alpha, \lambda, \mu$ 등의 에너지 함수 파라미터와 $\rho_{cls}$ 값에 따라 성능 차이가 발생하므로, 이에 대한 최적화 과정이 필수적이다.

## 📌 TL;DR

이 논문은 항공 이미지의 복잡한 배경과 객체 특성으로 인해 기존의 바운딩 박스 기반 인스턴스 분할 방법들이 겪는 노이즈 문제를 해결하기 위해, **Deep Level Set 기반의 곡선 진화 방식**을 제안하였다. 이를 통해 픽셀 수준의 정답 마스크 없이 바운딩 박스만으로도 정밀한 경계 복원이 가능함을 보였으며, iSAID와 Potsdam 데이터셋에서 기존 SOTA Box-supervised 방법론들을 뛰어넘는 성능을 달성하였다. 이 연구는 약한 감독(Weak supervision) 환경에서도 완전 감독(Full supervision)에 근접하는 성능을 낼 수 있는 가능성을 제시했다는 점에서 향후 연구에 중요한 역할을 할 것으로 보인다.