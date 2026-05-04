# Dual-Task Mutual Learning for Semi-Supervised Medical Image Segmentation

Yichi Zhang and Jicong Zhang (2021)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델의 성능을 높이기 위해서는 대량의 라벨링된 데이터가 필수적이다. 그러나 의료 영상의 특성상 정확한 어노테이션(Annotation)을 생성하기 위해서는 전문 의료진의 지식이 필요하며, 이는 매우 많은 비용과 시간이 소요되는 작업이다.

이러한 문제를 해결하기 위해 상대적으로 획득이 쉬운 라벨링되지 않은 데이터(Unlabeled data)를 활용하는 준지도 학습(Semi-supervised learning)이 주목받고 있다. 본 논문의 목표는 라벨링된 데이터가 극도로 부족한 상황에서도 unlabeled data를 효과적으로 활용하여 분할 성능을 극대화하는 **Dual-Task Mutual Learning (DTML)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 서로 다른 두 가지 태스크를 수행하는 두 개의 네트워크를 구성하고, 이들이 서로의 지식을 주고받게 하는 **상호 학습(Mutual Learning)** 구조를 도입하는 것이다.

1. **Dual-Task 설계**: 단순히 동일한 태스크를 수행하는 두 네트워크가 아니라, 하나는 영역 기반의 분할(Region-based segmentation)을, 다른 하나는 경계 기반의 거리 회귀(Boundary-based regression)를 학습하도록 설계하여 서로 다른 관점의 표현(Representation)을 학습하게 한다.
2. **기하학적 형상 제약(Geometric Shape Constraint) 부여**: Signed Distance Maps(SDM)를 학습 과정에 도입함으로써, 모델이 대상 객체의 단순한 픽셀 분류를 넘어 기하학적인 형상 정보를 학습하도록 강제한다.
3. **교차 태스크 일관성(Cross-task Consistency)**: 서로 다른 태스크의 출력물을 동일한 공간(Segmentation map)으로 매핑하여 두 네트워크 간의 예측 일관성을 유지하게 함으로써, unlabeled data로부터 유용한 정보를 효율적으로 추출한다.

## 📎 Related Works

### 준지도 의료 영상 분할 (Semi-Supervised Medical Image Segmentation)
기존 연구들은 주로 다음과 같은 두 가지 접근 방식을 사용했다.
- **의사 라벨링(Pseudo-labeling)**: 모델이 예측한 결과를 라벨로 사용하여 다시 학습하는 방식이다. 하지만 모델이 생성한 의사 라벨에 노이즈가 섞여 있을 경우, 학습 과정에서 부정적인 영향을 미칠 수 있다는 한계가 있다.
- **일관성 규제(Consistency Regularization)**: 입력 데이터에 작은 섭동(Perturbation)을 가했을 때 모델의 출력이 일정하게 유지되도록 하는 방식이다. 최근에는 Mean Teacher 구조나 적대적 학습(Adversarial learning)을 통해 분포의 유사성을 강제하는 방법들이 연구되었다.

### Signed Distance Maps (SDM)
일반적인 이진 마스크(Binary mask)와 달리, SDM은 각 픽셀에서 가장 가까운 경계선까지의 거리를 값으로 가지는 회색조 이미지이다. 이는 정답지(Ground Truth)에 대한 암시적 표현을 제공하여 모델이 경계선을 더 정밀하게 학습하도록 돕는다. 기존 연구들은 주로 보조 헤드(Auxiliary head)를 추가하는 방식으로 SDM을 활용했다.

## 🛠️ Methodology

### 전체 시스템 구조
본 프레임워크는 동일한 백본(Backbone) 구조를 공유하는 두 개의 개별 네트워크 $M_s$와 $M_d$로 구성된다.
- **$M_s$ (Segmentation Network)**: 입력 영상으로부터 분할 확률 맵(Segmentation probabilistic maps) $\hat{Y}_{seg}$를 생성한다.
- **$M_d$ (Regression Network)**: 입력 영상으로부터 Signed Distance Map(SDM) $\hat{Y}_{dis}$를 회귀(Regression)한다.

### 주요 구성 요소 및 수식 설명

**1. Signed Distance Map (SDM) 정의**
SDM $\text{G}_{\text{SDF}}$는 픽셀 $x$와 경계 $\partial\text{G}$ 사이의 유클리드 거리를 기반으로 정의된다.
$$
\text{G}_{\text{SDF}} = 
\begin{cases} 
-\inf_{y\in\partial\text{G}} \|x-y\|_2, & x \in \text{G}_{\text{in}} \\
0, & x \in \partial\text{G} \\
+\inf_{y\in\partial\text{G}} \|x-y\|_2, & x \in \text{G}_{\text{out}} 
\end{cases}
$$
여기서 객체 내부($\text{G}_{\text{in}}$)는 음수 값, 외부($\text{G}_{\text{out}}$)는 양수 값을 가지며, 절대값은 경계까지의 거리를 나타낸다.

**2. SDM의 마스크 변환**
$M_d$가 출력한 SDM을 분할 맵(Segmentation map) 형태로 변환하기 위해 다음과 같은 부드러운 근사 역변환(Smooth approximation to the inverse transform)을 사용한다.
$$
\text{G}_{\text{mask}} = \frac{1}{1 + e^{-k \cdot z}}, \quad z \in \text{G}_{\text{SDF}}
$$
여기서 $z$는 SDM의 값이며, $k$는 변환 계수이다.

**3. 손실 함수 및 학습 절차**
- **Supervised Loss (라벨 데이터 대상)**:
    - $M_s$는 Dice Loss와 Cross-Entropy Loss의 조합인 $L_{seg}$를 통해 학습한다.
    - $M_d$는 변환된 마스크와 정답 마스크 사이의 Dice Loss인 $L_{mask}$를 통해 학습한다. (실험적으로 $L_2$ 거리 손실보다 성능이 우수함이 확인되었다.)

- **Unsupervised Cross-Task Consistency Loss (비라벨 데이터 대상)**:
    unlabeled data에 대해 $M_s$의 예측값과 $M_d$의 예측값(변환 후)이 일치하도록 강제한다.
    $$
    L_{con} = \lambda_{con} \| f_{seg}(X; \theta_{seg}) - f_{mask}^{-1}(f_{dis}(X; \theta_{dis})) \|^2
    $$
    여기서 $\lambda_{con}$은 학습 횟수가 증가함에 따라 값이 커지는 Gaussian ramp-up 함수를 사용하여, 초기에는 지도 학습에 집중하고 점차 일관성 학습의 비중을 높인다.

- **최종 최적화 목표**:
    $M_s$와 $M_d$는 각각 다음의 목적 함수를 최소화하도록 학습된다.
    $$
    \min_{\theta_{seg}} \sum_{i \in \mathcal{D}^L} L_{seg}(f_{seg}(X_i; \theta_{seg}), Y_i) + \sum_{i \in \mathcal{D}^U} L_{con}
    $$
    $$
    \min_{\theta_{dis}} \sum_{i \in \mathcal{D}^L} L_{mask}(f_{mask}^{-1}(f_{dis}(X_i; \theta_{dis})), Y_i) + \sum_{i \in \mathcal{D}^U} L_{con}
    $$

## 📊 Results

### 실험 설정
- **데이터셋**: Atrial Segmentation Challenge의 좌심방(Left Atrium, LA) 데이터셋 (100개의 3D GE-MRI 스캔).
- **데이터 분할**: 학습 데이터 80개(라벨 16개, 비라벨 64개), 테스트 데이터 20개.
- **백본**: V-Net.
- **평가 지표**: Dice Similarity Coefficient (Dice), Jaccard Index, Average Surface Distance (ASD), 95% Hausdorff Distance (95HD).

### 주요 결과
- **Ablation Study**: $M_s$ 단독 또는 $M_d$ 단독 학습보다 DTML 프레임워크를 적용했을 때 모든 지표에서 유의미한 성능 향상이 나타났다 (p < 0.05).
- **정량적 비교**: 최신 준지도 학습 방법들(ASD-Net, TCSE, UA-MT, DTC, SASS, DoubleUnc)과 비교했을 때, 제안 방법이 가장 높은 성능을 보였다.
    - **Dice 결과**: 제안 방법(90.12%) $\gg$ DoubleUnc(89.65%) $>$ SASS(89.54%).
    - 특히, 모든 데이터를 사용한 Full-supervised (Upper bound) 성능인 91.14%에 근접하는 결과를 달성하였다.

## 🧠 Insights & Discussion

### 강점
본 연구의 핵심 강점은 서로 다른 성격의 두 태스크(분할 vs 회귀)를 결합하여 상호 학습을 유도했다는 점이다. 특히 SDM을 통해 경계 정보를 명시적으로 학습하게 함으로써, 단순한 픽셀 분류 모델이 놓치기 쉬운 기하학적 형상 제약을 효과적으로 부여하였다.

### 한계 및 논의
- **백본 의존성**: 모든 실험이 V-Net 기반으로 수행되었으므로, 다른 아키텍처(예: UNet++, Swin-UNETR 등)에서도 동일한 성능 향상이 나타날지는 추가 검증이 필요하다.
- **가정**: SDM을 마스크로 변환하는 과정에서 사용되는 계수 $k$의 설정이 결과에 영향을 줄 수 있으나, 이에 대한 최적화 방법론은 상세히 다뤄지지 않았다.
- **비판적 해석**: $M_d$의 학습 시 $L_{dis}$($L_2$ loss)보다 $L_{mask}$가 더 효과적이었다는 점은, 결국 SDM 자체의 정밀한 값보다는 그것이 암시하는 '경계의 위치'가 분할 성능에 더 결정적임을 시사한다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 라벨링 데이터 부족 문제를 해결하기 위해, **분할 맵 생성($M_s$)**과 **거리 맵 회귀($M_d$)**라는 두 가지 태스크를 상호 학습시키는 **DTML 프레임워크**를 제안한다. 특히 Signed Distance Maps를 활용한 기하학적 제약과 교차 태스크 일관성 손실을 통해 unlabeled data를 효과적으로 활용하였으며, 좌심방 분할 작업에서 기존 SOTA 준지도 학습 모델들을 뛰어넘는 성능을 입증하였다. 이 연구는 서로 다른 태스크의 시너지를 통해 데이터 효율성을 높이는 새로운 방향성을 제시한다.