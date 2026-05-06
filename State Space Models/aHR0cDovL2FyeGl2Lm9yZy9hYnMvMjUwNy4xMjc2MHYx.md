# Unified Medical Image Segmentation with State Space Modeling Snake

Ruicheng Zhang et al. (2025)

## 🧩 Problem to Solve

본 논문은 **Unified Medical Image Segmentation (UMIS)**, 즉 의료 영상 내의 모든 관심 영역(ROI)을 통합적으로 분할하는 문제를 해결하고자 한다. UMIS는 조직의 수, 모양, 크기 또는 영상 모달리티에 관계없이 정확하게 분할해야 하므로 임상적 종합 진단에 매우 중요하다.

그러나 UMIS는 다음과 같은 **다중 척도 구조적 이질성(multi-scale structural heterogeneity)**으로 인해 매우 까다로운 과제이다.

- **경계 모호성:** 영상 품질 저하 및 장기 간의 근접성으로 인해 경계가 흐릿하거나 겹치는 현상이 발생한다.
- **형태적 변이:** 해부학적 구조가 공간적 척도에 따라 중첩된 형태적 변이(모양, 크기, 위치, 방향 등)를 보이며, 특히 병변으로 인한 기형적 변형이 발생할 수 있다.
- **특성 충돌:** 거대 장기는 저주파 특성에, 미세 구조는 고주파 세부 사항에 의존하므로, 이를 동시에 학습할 때 특성 간의 간섭이 발생하여 미세 구조가 과소 분할(under-segmentation)되는 경향이 있다.

기존의 픽셀 기반(pixel-based) 방식은 객체 수준의 해부학적 통찰력과 장기 간의 관계 모델링 능력이 부족하여, 위상적 일관성이 결여되거나 형태적 변이에 취약하다는 한계가 있다.

## ✨ Key Contributions

본 논문은 상태 공간 모델(State Space Modeling)을 결합한 새로운 딥 스네이크(deep snake) 프레임워크인 **Mamba Snake**를 제안한다. 핵심 아이디어는 다중 윤곽선 진화(multi-contour evolution) 과정을 **계층적 상태 공간 아틀라스(hierarchical state space atlas)**로 모델링하는 것이다.

주요 기여 사항은 다음과 같다.

- **계층적 상태 공간 아틀라스:** 거시적(macroscopic) 관점에서는 장기 간의 위상적 관계를 모델링하고, 미시적(microscopic) 관점에서는 개별 장기의 윤곽선 진화에 집중한다.
- **Mamba Evolution Block (MEB):** 스네이크 전용 비전 상태 공간 모듈을 설계하여, 인과적 제약(causal constraints) 없이 주변 포인트의 정보를 통합하고 과거의 진화 이력을 기억하는 시공간 메모리 메커니즘을 구현하였다.
- **에너지 형상 사전 정보(Energy Shape Prior Map, ESPM):** 거리 변환(distance transform) 기반의 에너지 맵을 통해 윤곽선 진화에 지속적인 해부학적 가이드를 제공하여 초기 설정에 대한 민감도를 낮추고 강건성을 높였다.
- **이중 분류 시너지(Dual-Classification Synergy):** 검출(Detection)과 분할(Segmentation)을 동시에 최적화하는 메커니즘을 통해 미세 구조의 과소 분할 문제를 완화하고 오차 전파를 줄였다.

## 📎 Related Works

### 1. Unified Medical Image Segmentation

최근 SAM(Segment Anything Model)을 의료 분야에 적응시킨 SAM-Med2D, MedSAM 등이 제안되었으며, 지속적 학습이나 텍스트 임베딩을 활용한 방법들이 연구되었다. 하지만 이러한 픽셀 기반 모델들은 경계 모호성 해결 능력이 부족하고, 모델 파라미터 수가 너무 많아 실제 임상 배포에 어려움이 있다.

### 2. Deep Snake Model

전통적인 Active Contour Model을 딥러닝과 결합하여 객체 수준의 윤곽선을 예측하는 방식이다. 픽셀 기반 방식보다 매끄럽고 정확한 윤곽선을 생성하며 위상적 일관성을 유지한다는 장점이 있다. 그러나 초기 검출 단계의 오류가 전파되는 문제, 진화 과정에서 동적 특성과 과거 이력을 무시하여 윤곽선이 과하게 단순화(over-smoothed)되는 한계가 존재한다.

### 3. State Space Model (SSM)

Mamba와 같은 SSM은 선형 복잡도로 전역 수용 영역(global receptive field)을 제공한다. 하지만 기존 SSM은 인과적(causal) 특성으로 인해 윤곽선 포인트의 등방성(isotropic) 정보 통합이 어렵고, 반복적인 진화에 필요한 시간적 특성을 간과하는 경향이 있다.

## 🛠️ Methodology

Mamba Snake는 크게 **검출 단계(Detection Stage)**와 **진화 단계(Evolution Stage)**의 두 단계로 구성된다.

### 1. 에너지 형상 사전 정보 기반 진화 (ESPM)

학습 가능한 에너지 매핑을 통해 윤곽선 포인트들을 끌어당기는 연속적인 장(field)을 형성한다. 픽셀 레벨의 에너지 값 $E(x, y)$는 다음과 같이 정의된다.
$$E(x, y) = D_T(u) * G_\sigma + \lambda \|\nabla u\|^{-0.5}$$
여기서 $D_T(u)$는 예측된 조직 경계로부터의 거리 변환(distance transform), $G_\sigma$는 가우시안 스무딩, $\|\nabla u\|^{-0.5}$는 약한 경계에서의 그래디언트 응답을 증폭시키는 항이다. 거리 변환 함수로 Linear, Exponential, Logarithmic 함수를 검토하였으며, 이를 통해 초기 윤곽선 위치에 대한 민감도를 낮추고 장거리 가이드를 제공한다.

### 2. 상태 공간 메모리 역학 (SSMD)

다중 윤곽선 진화를 거시적 및 미시적 아틀라스로 나누어 처리한다.

- **거시적 아틀라스(Macroscopic Atlas):** 검출 박스 내에서 그리드 샘플링을 통해 특징 벡터를 추출하고, 상태 공간 변형 모델을 통해 초기 다각형의 정점 오프셋을 예측하여 장기 간의 위상적 관계를 모델링한다.
- **미시적 아틀라스(Microscopic Atlas):** 초기 다각형을 타겟 경계에 정밀하게 맞추기 위해 반복적인 정밀화 과정을 거친다. 각 정점 $x_i$에 대해 특징 벡터 $f_i$를 구성하고, 변형 모델 $\Psi$를 통해 오프셋 $\Delta x_i$를 예측하여 위치를 업데이트한다.
  $$x'_i = x_i + \Delta x_i$$

#### Mamba Evolution Block (MEB)

MEB는 기존 SSM의 인과적 제약을 깨고 윤곽선 포인트 간의 효율적인 상호작용을 가능하게 한다.

- **Circular Convolution:** 정점들이 위상적으로 인접해 있다는 점을 이용하여 원형 컨볼루션을 통해 주변 포인트의 공간 정보를 집계한다.
- **Temporal Memory:** 과거의 은닉 상태(hidden state)를 유지하여 현재의 진화 방향을 결정한다.
- **수식:**
  $$Z_i = \delta_i A_i (Z_{i-1} + B_i X_i), \quad Y_i = C_i^\top Z_i + D_i X_i$$
  여기서 $Z_i$는 시공간적 특징을 캡처하는 은닉 상태이며, $\delta_i$는 현재와 과거의 기여도를 조절하는 가중치이다.

### 3. 이중 분류 시너지 (Dual-Classification Synergy)

검출 분류기 $C_{det}$와 분할 분류기 $C_{seg}$를 동시에 사용하여 상호 보완적인 학습을 수행한다.

- **손실 함수:**
  $$\mathcal{L}_{ce} = -\sum_{c=1}^C y_c \log(\text{softmax}(\omega_{det} p_{det} + \omega_{seg} p_{seg})_c)$$
  $$\mathcal{L}_{con} = \lambda \left( -\sum_{c=1}^C \text{softmax}(p_{det})_c \log(\text{softmax}(p_{seg})_c) \right)$$
  여기서 $\mathcal{L}_{con}$은 두 분류기 간의 일관성을 강제하는 일관성 손실(consistency loss)이며, 특히 작은 크기의 장기에 대해 가중치 $\lambda$를 높게 설정하여 과소 분할 문제를 해결한다.

### 4. 학습 및 구현 세부 사항

- **에너지 맵 생성:** EfficientNetV2-S 백본과 디코더를 사용하며, Charbonnier 손실 함수로 사전 학습한다.
- **검출기:** CenterNet을 사용하여 초기 바운딩 박스를 생성한다.
- **윤곽선 진화:** 128개의 정점을 사용하며, 총 3회의 반복 진화를 수행한다.
- **전체 손실 함수:** $\mathcal{L} = \mathcal{L}_{ex} + \mathcal{L}_{evol} + 0.5\mathcal{L}_{ce} + 0.5\mathcal{L}_{con} + \mathcal{L}_{detector}$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** MR_AVBCE(척추 MRI), VerSe(척추 CT), RAOS(복부 CT), PanNuke(세포 현미경), BTCV(복부 CT) 등 5개의 임상 데이터셋을 사용하였다.
- **평가 지표:** mIoU, mDice, mBoundF(경계 정밀도 측정 지표)를 사용하였다.

### 2. 정량적 결과

Mamba Snake는 모든 데이터셋에서 SOTA 모델들을 압도하였다.

- **평균 성능:** 기존 최신 방법론 대비 평균 Dice 계수가 약 **3% 향상**되었다.
- **경계 정밀도:** 특히 mBoundF 지표에서 큰 폭의 상승을 보였는데, 이는 딥 스네이크 알고리즘의 특성과 Mamba 기반의 정밀한 진화 전략이 결합된 결과로 해석된다.
- **미세 구조 성능:** MR_AVBCE 데이터셋의 26개 세부 구조 중 상당수에서 가장 높은 성능을 기록하였다.

### 3. 정성적 결과

시각화 결과, 픽셀 기반 방법들이 인접한 장기를 구분하지 못하거나 마스크 내부에 빈 공간(void)을 만드는 것과 달리, Mamba Snake는 객체 수준의 경계를 매끄럽고 정확하게 획득하였다. 특히 흐릿한 추간판(inter-vertebral discs)이나 작은 장기들에서도 우수한 분할 능력을 보여주었다.

### 4. 절제 연구 (Ablation Study)

- **ESPM:** mDice와 mIoU를 약 3% 향상시키며, 흐릿한 경계에 대한 강건성을 부여한다.
- **SSMD:** 모든 지표에서 4% 이상의 성능 향상을 보였으며, 특히 장기 간 위상 관계 모델링을 통해 윤곽선 겹침 현상을 줄였다.
- **DCS:** 특히 mBoundF의 향상이 두드러졌으며, 작은 장기의 과소 분할을 **47% 감소**시켰다.

## 🧠 Insights & Discussion

### 강점

본 논문은 픽셀 단위의 예측에서 벗어나 **객체 수준의 윤곽선 진화**라는 관점으로 접근하여 의료 영상의 고질적인 문제인 경계 모호성과 위상적 불일치 문제를 효과적으로 해결하였다. 특히 Mamba의 선형 복잡도와 전역 수용 영역을 윤곽선 포인트의 시공간적 메모리로 치환한 점이 매우 독창적이다.

### 한계 및 미해결 질문

논문에서 명시적으로 언급된 한계점은 다음과 같다.

1. **내부 구멍(Holes) 처리:** 윤곽선 기반 방식의 특성상 객체 내부에 구멍이 있는 구조를 처리하는 데 어려움이 있다.
2. **극소 구조 및 단절된 구조:** 단 몇 픽셀 수준의 매우 작은 객체나 위상적으로 분리된 경계를 가진 구조에서는 픽셀 기반 방식보다 성능이 떨어질 수 있다.
3. **검출 의존성:** 2단계 파이프라인이므로, 초기 검출 단계에서 객체를 찾지 못하면 진화 단계 자체가 시작될 수 없다.

### 비판적 해석

Mamba Snake는 경계 정밀도에서 압도적인 성능을 보이지만, 이는 결국 '초기 검출'이라는 전제 조건에 강하게 결합되어 있다. 따라서 검출기의 성능이 전체 시스템의 상한선(Upper bound)이 될 위험이 있다. 또한, 3D 의료 영상을 2D 슬라이스로 변환하여 처리하므로, 슬라이스 간의 연속성(inter-slice consistency)을 활용하는 3D SSM으로의 확장이 향후 중요한 연구 방향이 될 것으로 보인다.

## 📌 TL;DR

Mamba Snake는 **상태 공간 모델(SSM)을 딥 스네이크 프레임워크에 통합**하여 의료 영상의 다중 척도 구조적 이질성 문제를 해결한 모델이다. 계층적 아틀라스 구조와 Mamba Evolution Block을 통해 장기 간의 위상 관계를 보존하면서도 정밀한 경계 획득이 가능하며, 에너지 사전 정보와 이중 분류 메커니즘으로 강건성을 높였다. 결과적으로 5개 데이터셋에서 기존 SOTA 대비 평균 3%의 Dice 향상을 달성하였으며, 특히 정밀한 경계 획득이 필수적인 임상 진단 도구로서 높은 잠재력을 가진다.
