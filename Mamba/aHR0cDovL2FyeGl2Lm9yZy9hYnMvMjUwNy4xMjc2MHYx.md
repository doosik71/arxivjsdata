# Unified Medical Image Segmentation with State Space Modeling Snake

Ruicheng Zhang, Haowei Guo, Kanghui Tian, Jun Zhou, Mingliang Yan, Zeyu Zhang, Shen Zhao (2025)

## 🧩 Problem to Solve

본 논문은 통합 의료 영상 분할(Unified Medical Image Segmentation, UMIS)에서 발생하는 **다중 스케일 구조적 이질성(multi-scale structural heterogeneity)** 문제를 해결하고자 한다. UMIS는 의료 영상 내의 모든 관심 영역(ROI)의 경계를 정밀하게 묘사하는 것을 목표로 하며, 이는 암 진단이나 방사선 치료 계획과 같은 포괄적인 해부학적 평가에 필수적이다.

그러나 UMIS는 다음과 같은 세 가지 주요 난제에 직면해 있다:
1. **경계 모호성**: 영상 품질 저하나 장기 간의 근접성 및 중첩으로 인해 경계가 흐릿해지는 현상이 발생한다.
2. **형태적 변이**: 해부학적 구조가 공간적 스케일에 따라 중첩된 형태적 변이(크기, 모양, 위치, 방향 등)를 보이며, 특히 병변으로 인한 국소적 변형이 전체적인 기하학적 구조를 왜곡한다.
3. **특성 충돌**: 거대 장기는 저주파 특성에, 미세 구조는 고주파 세부 사항에 의존한다. 이러한 스펙트럼 차이와 특성 간의 간섭은 특히 미세 구조의 과소 분할(under-segmentation) 문제를 야기한다.

기존의 픽셀 기반(pixel-based) 방식은 객체 수준의 해부학적 통찰과 장기 간의 관계 모델링이 부족하여 위와 같은 복잡한 형태적 변화에 취약하며, 경계가 불연속적이거나 부자연스러운 결과를 생성하는 한계가 있다.

## ✨ Key Contributions

본 논문은 상태 공간 모델링(State Space Modeling)을 결합한 새로운 딥 스네이크(Deep Snake) 프레임워크인 **Mamba Snake**를 제안한다. 핵심 아이디어는 다중 컨투어(contour) 진화 과정을 **계층적 상태 공간 아틀라스(hierarchical state space atlas)**로 모델링하여, 거시적인 장기 간 위상 관계와 미시적인 컨투어 정밀화를 동시에 달성하는 것이다.

주요 기여 사항은 다음과 같다:
- **Mamba Evolution Block (MEB)**: 컨투어 포인트의 진화를 위한 전용 시각 상태 공간 모듈을 설계하였다. 이는 원형 컨볼루션(circular convolution)을 통해 공간 정보를 통합하고, 시간적 은닉 상태(temporal hidden states)를 통해 진화의 동적 특성을 캡처한다.
- **에너지 형상 사전 정보(Energy Shape Prior Map, ESPM)**: 경계 거리 변환(boundary distance transform)을 이용한 에너지 맵을 도입하여, 초기 컨투어 배치에 대한 민감도를 낮추고 강건한 장거리 컨투어 진화를 유도한다.
- **이중 분류 시너지(Dual-Classification Synergy, DCS)**: 탐지(Detection)와 분할(Segmentation)을 동시에 최적화하는 메커니즘을 통해 미세 구조의 검출 및 분할 성능을 높이고 에러 전파를 억제한다.

## 📎 Related Works

### 1. 통합 의료 영상 분할 (UMIS)
최근 SAM(Segment Anything Model)을 의료 분야에 적응시킨 연구들이 등장하며 일반화 성능을 높였으나, 여전히 경계 모호성 해결과 미세 구조 분할에는 한계가 있다. 또한, 모델의 파라미터 수가 방대하여 실제 임상 환경에 배치하기 어렵다는 단점이 있다.

### 2. 딥 스네이크 모델 (Deep Snake Model)
전통적인 능동 컨투어 모델(Active Contour Model)을 딥러닝과 결합한 방식으로, 픽셀 기반 방식보다 매끄럽고 정확한 객체 수준의 경계를 생성할 수 있다. 하지만 초기 탐지 단계의 오류가 이후 진화 단계로 전파되는 문제와, 컨투어 변형의 동적 특성 및 이력을 간과하여 결과적으로 너무 매끄럽게만 처리되는(over-smoothed) 경향이 있다.

### 3. 상태 공간 모델 (State Space Model, SSM)
Mamba와 같은 SSM은 선형 복잡도로 전역 수용 영역(global receptive field)을 제공한다. 그러나 기존 SSM은 인과적(causal) 특성으로 인해 주변 포인트 정보를 등방성(isotropic)으로 집계하는 데 어려움이 있으며, 반복적인 컨투어 진화에 필요한 시간적 특성을 충분히 고려하지 않는다.

## 🛠️ Methodology

Mamba Snake는 크게 **탐지 단계(Detection Stage)**와 **진화 단계(Evolution Stage)**의 두 단계 파이프라인으로 구성된다.

### 1. Energy Shape Prior Map (ESPM)
컨투어 포인트의 좌표를 규제하기 위해 학습 가능한 에너지 맵을 생성한다. 픽셀 레벨의 에너지 값 $\mathcal{E}(x,y)$는 다음과 같이 정의된다:
$$\mathcal{E}(x,y) = D^T(\mathcal{U}) * \mathcal{G}_\sigma + \gamma \|\nabla \mathcal{U}\|^{-0.5}$$
여기서 $D^T(\mathcal{U})$는 예측된 조직 경계로부터의 거리 변환, $\mathcal{G}_\sigma$는 가우시안 스무딩, $\|\nabla \mathcal{U}\|^{-0.5}$는 약한 경계에서의 그래디언트 응답을 증폭시키는 항이다. 이 ESPM은 컨투어 진화 과정에서 장거리 가이드를 제공하여 초기화 위치에 관계없이 견고하게 수렴하도록 돕는다.

### 2. State Space Memory Dynamics (SSMD)
컨투어 진화를 거시적(Macroscopic) 관점과 미시적(Microscopic) 관점으로 나누어 처리한다.

- **거시적 아틀라스 진화**: 탐지 박스 내에서 희소하게 샘플링된 그리드 포인트들을 이용해 장기 간의 위상 관계 및 해부학적 계층 구조를 모델링하고 초기 폴리곤을 생성한다.
- **미시적 아틀라스 진화**: 생성된 폴리곤의 각 정점(vertex)에 대해 반복적인 정밀화를 수행한다. 정점 $x_i$는 다음과 같이 업데이트된다:
$$x'_i = x_i + \Delta x_i$$
여기서 $\Delta x_i$는 상태 공간 변형 모델 $\Psi$에 의해 예측된 오프셋이다.

#### Mamba Evolution Block (MEB)
MEB는 기존 SSM의 인과적 제약을 깨고 비인과적(non-causal) 프레임워크를 구축한다. 원형 컨볼루션을 사용하여 인접한 포인트 간의 공간 정보를 집계하며, 과거의 은닉 상태를 유지하여 시간적 특성을 학습한다. 수식은 다음과 같다:
$$Z_i = \delta_i A_i (Z_{i-1} + B_i X_i)$$
$$Y_i = C_i^\top Z_i + D_i X_i$$
여기서 $Z_i$는 상태 공간을 인코딩하는 잠재 변수, $A_i$는 전이 행렬(스칼라로 정의), $X_i$는 입력 토큰이며, $\delta_i$는 현재와 과거의 기여도를 조절하는 가중치이다.

### 3. Dual-Classification Synergy (DCS)
탐지 분류기 $\mathcal{C}_{det}$와 분할 분류기 $\mathcal{C}_{seg}$를 통해 상호 보완적인 학습을 수행한다.
- **교차 엔트로피 손실 $\mathcal{L}_{ce}$**: 두 분류기의 예측 확률 $p_{det}$와 $p_{seg}$의 가중 평균을 기반으로 계산한다.
- **일관성 손실 $\mathcal{L}_{con}$**: 두 예측 확률 분포 간의 정렬을 강제한다.
$$\mathcal{L}_{con} = \lambda \left( -\sum_{c=1}^C \text{softmax}(p_{det})_c \log(\text{softmax}(p_{seg})_c) \right)$$
여기서 $\lambda$는 마스크 크기에 반비례하는 페널티 팩터로, 작은 장기의 검출 및 분할 성능을 높이는 역할을 한다.

### 4. 학습 절차 및 손실 함수
1. **사전 학습**: EfficientNetV2-S 기반의 에너지 맵 생성 네트워크를 Charbonnier 손실 함수 $\mathcal{L}_{energy}$를 사용하여 먼저 학습시킨다.
2. **결합 학습**: 탐지기(CenterNet)와 컨투어 진화 네트워크를 엔드-투-엔드로 학습시킨다. 전체 손실 함수 $\mathcal{L}$은 다음과 같다:
$$\mathcal{L} = \mathcal{L}_{det} + \mathcal{L}_{evol} + 0.5\mathcal{L}_{ce} + 0.5\mathcal{L}_{con} + \mathcal{L}_{detector}$$
여기서 $\mathcal{L}_{evol}$은 예측된 컨투어 포인트와 실제 정답 포인트 간의 평균 $\ell_1$ 거리이다.

## 📊 Results

### 실험 설정
- **데이터셋**: MR_AVBCE(척추 MRI), VerSe(척추 CT), RAOS(복부 CT), PanNuke(세포 현미경), BTCV(복부 CT) 등 5개 데이터셋을 사용하였다.
- **평가 지표**: mIoU, mDice, mBoundF(경계 정밀도 측정 지표)를 사용하였다.
- **비교 대상**: 픽셀 기반 모델(U-Net, nnU-Net, UNETR, MedSAM 등) 및 컨투어 기반 모델(Deep Snake, ADMIRE)과 비교하였다.

### 주요 결과
- **정량적 성능**: 모든 데이터셋에서 SOTA 성능을 달성하였다. 특히 가장 어려운 MR_AVBCE 데이터셋에서 mDice가 기존 최고 성능 대비 약 3% 향상되었으며, mBoundF에서 매우 큰 폭의 개선을 보여 경계 묘사 능력이 탁월함을 입증하였다.
- **정성적 결과**: 픽셀 기반 방식이 흔히 겪는 픽셀 오분류, 마스크 공백, 불연속적인 경계 문제가 Mamba Snake에서는 해결되었다. 특히 종양으로 인한 변형이나 미세한 인터-버티브럴 디스크(inter-vertebral discs) 분할에서 우수한 성능을 보였다.
- **효율성**: 추론 속도와 계산 효율성 면에서도 경쟁력이 있음을 확인하였다.

### 절제 연구 (Ablation Study)
- **ESPM**: mDice와 mIoU를 약 3% 향상시키며, 흐릿한 경계와 복잡한 배경에 대한 내성을 높였다.
- **SSMD**: 모든 지표에서 4% 이상의 성능 향상을 보였으며, 이는 계층적 아틀라스 설계가 장기 간 위상 관계를 잘 캡처했기 때문이다.
- **DCS**: 특히 mBoundF 지표를 개선했으며, 미세 구조의 과소 분할 문제를 47% 감소시켰다.
- **하이퍼파라미터**: 진화 반복 횟수는 3회, 컨투어 포인트 수는 128개가 최적의 성능을 나타냈다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구의 핵심 성과는 픽셀 단위의 예측에서 벗어나 **객체 수준의 컨투어 진화** 방식을 채택하고, 이를 **상태 공간 모델(SSM)**의 동적 메모리 메커니즘으로 가속화했다는 점이다. 특히 SSM의 인과적 특성을 제거하고 원형 컨볼루션을 도입함으로써, 컨투어 포인트들이 전후방 정보를 모두 참조할 수 있게 하여 복잡한 해부학적 구조에서도 매끄러운 경계를 생성할 수 있었다.

### 한계 및 비판적 해석
논문에서 명시한 한계점은 다음과 같다:
1. **내부 구멍(Hole) 처리**: 컨투어 기반 방식의 특성상, 객체 내부에 구멍이 있는 구조를 처리하는 데 어려움이 있다.
2. **극소형/불연속 구조**: 단 몇 픽셀로 이루어진 매우 작은 객체나 위상적으로 분리된 경계를 가진 구조에서는 픽셀 기반 방식보다 성능이 떨어질 수 있다.
3. **탐지기 의존성**: 2단계 구조이므로, 첫 단계인 탐지기(Detector)가 객체를 찾지 못하면 진화 단계 자체가 시작되지 않아 분할에 실패한다.

비판적으로 볼 때, 탐지 단계의 오류를 줄이기 위해 학습 시 Ground-Truth 박스를 사용하는 기법을 썼으나, 실제 추론 시 탐지기의 성능이 전체 파이프라인의 상한선(Upper bound)으로 작용할 위험이 크다.

## 📌 TL;DR

Mamba Snake는 의료 영상의 다중 스케일 구조적 이질성 문제를 해결하기 위해 **상태 공간 모델(SSM)**을 통합한 딥 스네이크 프레임워크이다. **에너지 형상 사전 정보(ESPM)**, **Mamba 진화 블록(MEB)**, **이중 분류 시너지(DCS)**를 통해 경계 정밀도를 극대화했으며, 5개 임상 데이터셋에서 기존 SOTA 대비 평균 3%의 Dice 성능 향상을 이끌어냈다. 이 연구는 특히 정밀한 경계 묘사가 중요한 수술 계획이나 방사선 치료 분야에서 높은 활용 가능성을 가진다.