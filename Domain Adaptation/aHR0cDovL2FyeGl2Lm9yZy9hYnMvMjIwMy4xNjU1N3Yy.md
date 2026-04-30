# COSMOS: Cross-Modality Unsupervised Domain Adaptation for 3D Medical Image Segmentation based on Target-aware Domain Translation and Iterative Self-Training

Hyungseob Shin, Hyeongyu Kim, Sewon Kim, Yohan Jun, Taejoon Eo, Dosik Hwang (2022/2023)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation) 분야에서 전문가의 픽셀 단위 어노테이션(Annotation)을 획득하는 데 드는 막대한 비용과 시간 문제를 해결하는 것이다. 특히, 서로 다른 MRI 모달리티(Contrast) 간의 도메인 차이로 인해, 한 모달리티에서 학습된 모델을 다른 모달리티에 그대로 적용할 경우 성능이 급격히 저하되는 문제가 발생한다.

논문의 구체적인 목표는 조영제 증강 T1-가중 영상(Contrast-enhanced T1-weighted, ceT1)의 어노테이션 데이터를 활용하여, 라벨이 전혀 없는 고해상도 T2-가중 영상(High-resolution T2-weighted, hrT2)에서 전정신경초종(Vestibular Schwannoma, VS)과 와우(Cochlea)를 자동으로 분할하는 Unsupervised Domain Adaptation(UDA) 프레임워크를 구축하는 것이다. 이는 가돌리늄 주입의 부작용과 긴 촬영 시간이라는 단점이 있는 ceT1 스캔의 대체재로 hrT2 스캔을 활용 가능하게 함으로써 임상적 비용을 낮추는 데 중요한 의미를 갖는다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Target-aware Domain Translation**과 **Iterative Self-Training**을 결합하여 3D 의료 영상의 모달리티 간 간극을 효과적으로 메운 점이다.

1.  **Target-aware Domain Translation**: 단순한 이미지 변환을 넘어, 변환 네트워크의 인코더를 분할 네트워크(Segmentor)와 공유함으로써 변환된 가상 이미지(Pseudo-image) 내에서 VS와 와우 같은 주요 해부학적 구조의 기하학적 형태가 보존되도록 강제한다.
2.  **Iterative Self-Training**: 가상 데이터로 학습된 초기 모델을 통해 실제 타겟 데이터의 의사 라벨(Pseudo-label)을 생성하고, 이를 다시 학습에 활용하는 과정을 반복함으로써 타겟 도메인의 데이터 분포에 모델을 점진적으로 적응시킨다.
3.  **3D UDA 검증**: 기존의 많은 UDA 연구가 2D 분할에 치중했던 것과 달리, 3D 의료 영상 분할 작업에서 본 방법론의 유효성을 입증하였으며, 특히 크기가 작은 다중 클래스 구조물 분할이라는 도전적인 과제에서 SOTA 성능을 달성하였다.

## 📎 Related Works

기존의 의료 영상 UDA 연구들은 주로 픽셀 레벨의 적응(Pixel-level adaptation) 방식에 기반하며, 이미지 변환 네트워크와 분할 네트워크를 end-to-end로 학습시켜 소스 도메인을 타겟 도메인으로 변환한 영상을 학습에 사용한다. 일부 연구에서는 합성 영상과 실제 타겟 영상의 분할 결과물 사이의 적대적 학습(Adversarial training)을 통해 구조적 기하학을 보존하려 시도하였다.

그러나 기존 방식들은 다음과 같은 한계가 존재한다:
- 대부분 2D 분할에 집중되어 있어 3D 데이터의 공간적 맥락을 충분히 활용하지 못한다.
- 타겟 도메인의 실제 데이터 분포를 학습 과정에 직접적으로 통합하는 기법이 부족하여, 합성 영상과 실제 영상 사이의 잔여 도메인 간극(Domain gap)을 완전히 해결하지 못한다.
- 주로 복부나 심장과 같은 큰 구조물 분할에 집중되어 있어, VS나 와우처럼 크기가 작은 구조물에 대한 정밀한 분할 성능은 검증되지 않았다.

## 🛠️ Methodology

COSMOS 프레임워크는 크게 두 단계로 구성된다.

### 1. Target-aware Domain Translation Network
소스 도메인($\text{ceT1}$)의 영상 $x_s$와 라벨 $y_s$, 그리고 라벨이 없는 타겟 도메인($\text{hrT2}$) 영상 $x_t$를 사용하여 CycleGAN 기반의 변환 네트워크를 학습한다. 핵심은 변환 디코더($G^d_D$)와 분할 디코더($G^d_S$)가 동일한 인코더($G^d_E$)를 공유하는 구조이다.

전체 손실 함수는 다음과 같이 정의된다:
$$\mathcal{L}_{cosmos} = \lambda_1 \mathcal{L}_{cycle} + \lambda_2 \mathcal{L}_{adv} + \lambda_3 \mathcal{L}_{identity} + \lambda_4 \mathcal{L}_{seg}$$

각 항의 상세 설명은 다음과 같다:
- **Cycle-consistency Loss ($\mathcal{L}_{cycle}$)**: $x_s \rightarrow \tilde{x}_t \rightarrow \hat{x}_s$ 과정을 통해 원래 영상으로 복원될 때의 $L1$ 거리를 측정하여 구조적 일관성을 유지한다.
- **Adversarial Loss ($\mathcal{L}_{adv}$)**: 생성된 영상이 실제 타겟 도메인의 분포와 유사해지도록 판별기($D_S, D_T$)를 통해 학습한다.
- **Identity Loss ($\mathcal{L}_{identity}$)**: 입력 영상이 이미 타겟 도메인일 경우 변환 없이 그대로 유지하도록 하여 색상 구성을 보존한다.
- **Segmentor Loss ($\mathcal{L}_{seg}$)**: 소스 도메인 라벨 $y_s$를 사용하여 분할 정확도를 높인다. 특히 소스-타겟 경로와 타겟-소스 경로 모두에서 분할을 수행함으로써 인코더가 주요 관심 구조(VS, 와우)의 특징에 집중하게 만든다.
$$\mathcal{L}_{seg} = (1 - \text{DSC}(y_s, \text{Seg}_{S \to T}(x_s))) + (1 - \text{DSC}(y_s, \text{Seg}_{T \to S}(C_{S \to T}(x_s))))$$

### 2. Iterative Self-Training
변환된 가상 T2 영상($\tilde{x}_t$)과 실제 T2 영상($x_t$) 사이의 잔여 간극을 줄이기 위해 nnU-Net을 백본으로 하는 자기 학습(Self-training)을 수행한다.

**절차:**
1.  **Teacher 모델 학습**: 라벨링된 가상 T2 데이터 $(\tilde{x}_t, y_s)$만을 사용하여 초기 Teacher 모델 $f_{teacher}$를 학습한다.
2.  **의사 라벨(Pseudo-label) 생성**: 학습된 Teacher 모델에 라벨이 없는 실제 T2 영상 $x_t$를 입력하여 의사 라벨 $\tilde{y}^{(t,0)}$을 생성한다.
3.  **Student 모델 학습**: 가상 데이터 $(\tilde{x}_t, y_s)$와 의사 라벨링된 실제 데이터 $(x_t, \tilde{y}^{(t,0)})$를 결합한 데이터셋으로 Student 모델 $f_{student}$를 학습한다.
$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} l_{seg}(y_{s_i}, f_{student}(\tilde{x}_{t_i})) + \frac{1}{m} \sum_{i=1}^{m} l_{seg}(\tilde{y}^{(t,0)}_i, f_{student}(x_{t_i}))$$
4.  **반복(Iteration)**: Student 모델을 새로운 Teacher 모델로 설정하고 의사 라벨을 갱신하는 과정을 $K=3$회 반복하여 라벨의 품질을 점진적으로 높인다.

## 📊 Results

### 실험 설정
- **데이터셋**: 어노테이션된 $\text{ceT1}$ 105건, 라벨 없는 $\text{hrT2}$ 105건(학습용), 라벨 없는 $\text{hrT2}$ 32건(검증용).
- **평가 지표**: $\text{Dice coefficient}$ ($\uparrow$), $\text{Average Symmetric Surface Distance (ASSD)}$ ($\downarrow$).
- **비교 대상**: CrossMoDA 챌린지의 상위 3개 팀(Team A, B, C) 및 ablation study 그룹.

### 주요 결과
- **정량적 성과**: COSMOS는 $\text{VS}$에 대해 $\text{Dice } 0.871 \pm 0.063$, $\text{ASSD } 0.437 \pm 0.270$을, $\text{Cochlea}$에 대해 $\text{Dice } 0.842 \pm 0.020$, $\text{ASSD } 0.152 \pm 0.030$을 기록하며 챌린지 1위를 차지하였다.
- **Ablation Study**:
    - **Target-aware Translation의 효과**: 단순 CycleGAN보다 구조 보존 능력이 뛰어나며, 이는 $\text{DA w/o Seg.}$ 대비 $\text{DA w/ Seg.}$의 성능 향상으로 증명되었다.
    - **Self-training의 효과**: 가상 데이터만 사용했을 때보다 실제 데이터의 의사 라벨을 결합했을 때 성능이 크게 향상되었다.
    - **반복 학습의 효과**: Iteration 1 $\rightarrow$ 2 $\rightarrow$ 3으로 갈수록 Dice 점수가 점진적으로 상승하고 ASSD가 감소하는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 단순한 이미지 변환을 넘어 **'분할을 위한 변환'**이라는 관점을 도입하여, 생성된 가상 이미지의 해부학적 정확도를 높인 점이 매우 고무적이다. 또한, 의료 영상 분야에서 드문 Iterative Self-training을 적용하여 가상 데이터의 완벽한 라벨과 실제 데이터의 정확한 분포라는 두 마리 토끼를 모두 잡으려 노력하였다.

**한계 및 논의사항:**
- **데이터 일반화**: 현재는 단일 데이터셋에서만 검증되었으므로, 다양한 기관이나 다른 기기에서 수집된 데이터셋에서도 동일한 성능이 나오는지 확인하는 일반화 검증이 필요하다.
- **의사 라벨의 신뢰도**: 현재는 VS나 와우가 완전히 누락된 영상만을 필터링하는 이미지 레벨 필터링을 사용하고 있다. 저자는 향후 논의에서 픽셀 레벨의 신뢰도 기반 필터링(Pixel-level filtering)을 통해 더욱 정교한 의사 라벨링을 구현할 가능성을 제시하였다.

## 📌 TL;DR

본 논문은 $\text{ceT1}$ MRI 라벨을 활용하여 라벨이 없는 $\text{hrT2}$ MRI에서 VS와 와우를 분할하는 UDA 프레임워크인 **COSMOS**를 제안한다. 분할 네트워크와 공유 인코더를 가진 **Target-aware Domain Translation**으로 구조적 보존력을 높이고, **Iterative Self-training**으로 실제 타겟 데이터의 분포에 적응함으로써 CrossMoDA 챌린지 1위의 성능을 달성하였다. 이 연구는 라벨 획득 비용이 높은 3D 의료 영상 분할 작업에서 효과적인 도메인 적응 전략을 제시하였다는 점에서 가치가 크다.