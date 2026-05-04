# Weakly Supervised Instance Segmentation by Learning Annotation Consistent Instances

Aditya Arun, C.V. Jawahar, and M. Pawan Contribution (2020)

## 🧩 Problem to Solve

본 논문은 픽셀 수준의 정밀한 주석(pixel-wise annotation) 없이도 객체의 클래스와 마스크를 동시에 추정하는 **Weakly Supervised Instance Segmentation (WSIS)** 문제를 해결하고자 한다. 

일반적인 Instance Segmentation 모델은 학습을 위해 막대한 비용이 드는 픽셀 단위 마스크 데이터가 필요하다. 이를 해결하기 위해 이미지 수준의 레이블(image-level labels)이나 바운딩 박스(bounding boxes)와 같은 약한 지도 학습(weak supervision) 방식이 제안되어 왔다. 기존의 WSIS 접근 방식들은 주로 두 단계로 구성된다: (1) 약한 주석과 일치하는 Pseudo Label을 생성하는 모델, (2) 생성된 Pseudo Label을 정답으로 간주하여 학습하는 Instance Segmentation 모델.

그러나 기존 방식들은 다음과 같은 한계가 있다. 첫째, Pseudo Label 생성 과정에서 발생하는 내재적인 불확실성(uncertainty)을 고려하지 않고 단일한 Pseudo Label에 의존한다. 둘째, 두 모델 간의 일관된 학습 목적 함수(learning objective)가 부족하여, 단순히 단계별로 학습시키거나 분석이 어려운 반복 학습 방식을 사용한다. 따라서 본 논문의 목표는 Pseudo Label 생성의 불확실성을 명시적으로 모델링하고, 두 모델을 통합적으로 최적화할 수 있는 확률론적 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Pseudo Label 생성 과정과 최종 예측 과정을 각각 **Conditional Distribution**과 **Prediction Distribution**으로 정의하고, 두 분포 사이의 **Dissimilarity Coefficient**를 최소화함으로써 모델을 학습시키는 것이다.

중심적인 설계 아이디어는 다음과 같다.
1. **불확실성 모델링**: Conditional Distribution에 노이즈를 추가하여 다양한 Pseudo Label 샘플을 생성함으로써, 단일 레이블에 의존할 때 발생하는 위험을 줄이고 불확실성을 명시적으로 처리한다.
2. **Annotation Consistency 강화**: 단순한 클래스 점수뿐만 아니라, 객체의 경계를 고려하는 Pairwise term과 이미지 수준의 주석과 일치하도록 강제하는 Higher-order term을 도입하여 더 정확한 Pseudo Label을 생성한다.
3. **통합 학습 목적 함수**: 두 분포 간의 차이를 측정하는 Jensen difference 기반의 Dissimilarity Coefficient를 사용하여, 예측 모델이 조건부 모델의 분포를 따라가도록 유도하는 공동 학습 체계를 제안한다.

## 📎 Related Works

기존의 WSIS 연구들은 주로 Class Activation Maps (CAM)를 기반으로 객체의 위치를 찾고 이를 마스크로 확장하는 방식을 사용했다. 하지만 CAM 기반 방식들은 객체의 가장 변별력이 높은(discriminative) 부분만을 강조하는 경향이 있어, 객체 전체를 완전히 커버하지 못하는 한계가 있다.

최근에는 Pseudo Label을 생성하고 이를 통해 Supervised 모델(예: Mask R-CNN)을 학습시키는 2단계 방식이 주를 이룬다. 예를 들어, IRN [1]이나 WISE [23] 같은 연구들이 이에 해당한다. 그러나 이러한 방법들은 Pseudo Label 생성 과정의 불확실성을 모델링하지 않았으며, 두 모델을 연결하는 수학적으로 정밀한 통합 목적 함수가 부재했다. 본 논문은 잠재 변수 모델(latent variable model)에서 불확실성을 다루는 방식을 차용하여 이를 WSIS에 적용함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
본 프레임워크는 **Conditional Network**($\text{Pr}_c$)와 **Prediction Network**($\text{Pr}_p$)라는 두 개의 네트워크로 구성된다. Conditional Network는 약한 주석을 바탕으로 가능한 마스크 후보들을 샘플링하며, Prediction Network는 이미지로부터 직접 인스턴스 마스크를 예측한다.

### 2. Conditional Distribution ($\text{Pr}_c$)
이 모델은 이미지 $x$와 약한 주석 $a$가 주어졌을 때 Pseudo Label $y$의 분포를 모델링한다. 구현상으로는 modified U-Net 구조를 사용하며, 네트워크의 마지막 레이어에 균등 분포에서 샘플링된 노이즈 $z$를 추가하여 다양한 샘플을 생성한다.

최종 점수 함수 $S^k(y_c)$는 다음 세 가지 항의 합으로 구성된다.
- **Semantic Class Aware Unary Term**: 각 세그멘테이션 제안(proposal)이 특정 클래스에 속할 확률을 계산한다.
- **Boundary Aware Pairwise Term**: 유나리 항이 객체의 일부만 선택하는 경향을 막기 위해, 인접한 제안들이 경계 픽셀 값에 따라 유사한 점수를 갖도록 유도한다. 이는 다음 식을 통한 반복 업데이트로 구현된다.
  $$G_{u,y_u}^{k,n} = G_{u,y_u}^{k,n-1} + \frac{1}{H_{u,v}^{k,n-1} + \delta} \exp(-I_{u,v})$$
  여기서 $I_{u,v}$는 인접 제안 사이의 엣지 픽셀 합이며, 이를 통해 경계가 뚜렷하지 않은 영역의 점수를 동기화하여 객체 전체를 커버하게 한다.
- **Annotation Consistent Higher Order Term**: 이미지 수준 주석에서 클래스 $j$가 존재한다고 명시되었다면, 생성된 샘플 중 최소한 하나는 해당 클래스여야 한다는 전역 제약 조건을 부여한다. 이를 위해 조건을 만족하지 않을 경우 점수를 $-\infty$로 처리한다.

### 3. Prediction Distribution ($\text{Pr}_p$)
예측 모델로는 Mask R-CNN을 사용하며, 이는 입력 이미지 $x$에 대해 인스턴스 마스크 분포 $\text{Pr}_p(y|x; \theta_p)$를 출력한다. 이는 각 바운딩 박스 제안에 대해 독립적으로 예측이 이루어지는 fully factorized distribution으로 간주된다.

### 4. Learning Objective
본 논문은 두 분포 $\text{Pr}_p$와 $\text{Pr}_c$ 사이의 **Dissimilarity Coefficient ($\text{DISC}_\Delta$)**를 최소화하는 것을 목표로 한다.

- **Diversity ($\text{DIV}_\Delta$)**: 두 분포에서 무작위로 샘플을 뽑았을 때 발생하는 기대 손실이다.
  $$\text{DIV}_\Delta(\text{Pr}_1, \text{Pr}_2) = \mathbb{E}_{y_1 \sim \text{Pr}_1} [ \mathbb{E}_{y_2 \sim \text{Pr}_2} [ \Delta(y_1, y_2) ] ]$$
- **Dissimilarity Coefficient**: Jensen difference를 사용하여 다음과 같이 정의된다.
  $$\text{DISC}_\Delta(\text{Pr}_1, \text{Pr}_2) = \text{DIV}_\Delta(\text{Pr}_1, \text{Pr}_2) - \gamma \text{DIV}_\Delta(\text{Pr}_2, \text{Pr}_2) - (1-\gamma) \text{DIV}_\Delta(\text{Pr}_1, \text{Pr}_1)$$
  여기서 $\gamma=0.5$를 사용하여 대칭성을 확보한다.

**Task-Specific Loss ($\Delta$)**는 Mask R-CNN의 다중 작업 손실 함수를 사용한다:
$$\Delta(y_1, y_2) = \Delta^{cls}(y_1, y_2) + \Delta^{box}(y_1, y_2) + \Delta^{mask}(y_1, y_2)$$
Conditional Network는 바운딩 박스를 직접 예측하지 않으므로 $\Delta^{box}$ 항의 기울기는 0이 되며, 대신 생성된 Pseudo Label의 타이트한 바운딩 박스를 생성하여 Prediction Network의 학습에 활용한다.

### 5. 학습 절차 (Optimization)
**Block Coordinate Descent** 전략을 사용하여 $\theta_p$와 $\theta_c$를 교대로 최적화한다.
1. **Prediction Network 학습**: Conditional Network를 고정하고 $\text{DISC}_\Delta$를 최소화하여 $\theta_p$를 업데이트한다.
2. **Conditional Network 학습**: Prediction Network를 고정하고 $\theta_c$를 업데이트한다. 이때 $\text{argmax}$ 연산으로 인해 발생하는 미분 불가능 문제는 Direct Loss Minimization 전략을 통해 근사 기울기를 계산하여 해결한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Augmented PASCAL VOC 2012 (학습 이미지 10,582장).
- **지도 수준**: (1) 이미지 수준 레이블 전용, (2) 바운딩 박스 + 이미지 수준 레이블.
- **평가 지표**: 다양한 IoU 임계값($0.25, 0.5, 0.7, 0.75$)에서의 mAP.

### 주요 결과
- **정량적 성과**: 이미지 수준 주석만 사용했을 때 $\text{mAP}_{0.5}$에서 50.9%를 달성하여 기존 최고 성능 대비 4.2% 향상되었으며, $\text{mAP}_{0.75}$에서는 28.5%를 기록했다. 바운딩 박스 주석을 사용했을 때는 $\text{mAP}_{0.75}$에서 32.1%를 기록하여 기존 SOTA 대비 10% 이상 향상되었다.
- **경계 정확도**: 특히 높은 IoU 임계값($0.7, 0.75$)에서 성능 향상이 두드러지는데, 이는 Pairwise term 등이 객체의 경계를 더 정확하게 예측하도록 유도했기 때문이다.
- **아키텍처 비교**: Conditional Network로 U-Net을 사용한 것이 ResNet 기반 구조보다 일관되게 높은 성능을 보였다. 이는 U-Net의 인코더-디코더 구조가 더 고해상도의 특징을 유지하여 작은 객체나 복잡한 환경에서 유리하기 때문으로 분석된다.

### 절제 실험 (Ablation Study)
- **구성 요소의 영향**: Unary(U) $\rightarrow$ U+Pairwise(P) $\rightarrow$ U+P+Higher-order(H) 순으로 성능이 크게 향상되었다. 특히 Pairwise term은 la-bel이 객체의 가장 변별력 있는 부분에만 쏠리는 현상을 막아 전체 영역을 커버하게 함으로써 $\text{mAP}_{0.75}$ 성능을 크게 끌어올렸다.
- **확률론적 목적 함수의 중요성**: 단순한 Pointwise 네트워크(Self-diversity 항 제거)보다 제안된 확률론적 분포 모델링을 사용했을 때 성능이 더 높았다. 특히 Conditional Network의 self-diversity를 유지하는 것이 오버피팅 방지와 어려운 케이스 해결에 중요함이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 WSIS에서 Pseudo Label의 '단일성'과 '불확실성' 문제를 확률론적 분포 정렬(distribution alignment)이라는 관점에서 성공적으로 해결하였다.

**강점**으로는 단순히 더 좋은 Pseudo Label을 만드는 것에 그치지 않고, 두 네트워크가 서로를 가이드하며 함께 성장하는 통합 학습 체계를 구축했다는 점이다. 특히 Boundary aware pairwise term을 통해 CAM 기반 방식의 고질적인 문제인 '부분적 활성화(partial activation)' 문제를 수학적으로 완화한 점이 인상적이다.

**한계 및 논의사항**으로는, Conditional Network의 추론 과정에서 MCG(Multiscale Combinatorial Grouping)와 같은 외부 제안 알고리즘에 의존하고 있다는 점이다. 만약 제안된 영역(proposal) 자체가 부실할 경우, 이후의 최적화 과정에서도 이를 회복하기 어려울 가능성이 있다. 또한, 반복적인 Block Coordinate Descent 방식은 학습 시간이 증가하는 요인이 될 수 있다.

## 📌 TL;DR

본 연구는 **Conditional Distribution(Pseudo Label 생성)**과 **Prediction Distribution(최종 예측)**이라는 두 확률 분포를 정의하고, 이들 사이의 **Dissimilarity Coefficient를 최소화**함으로써 약한 지도 학습 기반의 인스턴스 세그멘테이션 성능을 극대화했다. 특히 경계 인식 항(Pairwise term)과 주석 일치 항(Higher-order term)을 통해 고정밀 마스크를 생성할 수 있게 하였으며, PASCAL VOC 2012 데이터셋에서 SOTA 성능을 달성했다. 이 연구는 불확실성이 큰 약한 지도 학습 환경에서 확률론적 정렬 방식이 매우 유효함을 입증하였으며, 향후 다른 약한 지도 학습 태스크로 확장될 가능성이 높다.