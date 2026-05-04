# Distillation Learning Guided by Image Reconstruction for One-Shot Medical Image Segmentation

Feng Zhou, Yanjie Zhou, Longjie Wang, Yun Peng, David E. Carlson, and Liyun Tu (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **One-Shot Medical Image Segmentation (MIS)**이다. 의료 영상 분야에서 3D 영상의 정밀한 세그멘테이션(Segmentation)을 위해서는 전문가의 수작업 라벨링이 필수적이지만, 이는 막대한 시간과 비용이 소요된다. 따라서 단 하나의 라벨링된 데이터(Atlas)만으로 새로운 영상의 영역을 구분하는 One-Shot 학습이 매우 중요하다.

기존의 One-Shot MIS 방법론들은 주로 두 가지 방향으로 접근해 왔다. 첫째는 Registration 네트워크를 통해 Atlas의 라벨을 대상 영상으로 전파하는 **Atlas-Based Segmentation (ABS)** 방식이며, 둘째는 Registration을 통해 합성된 라벨 데이터를 생성하여 세그멘테이션 네트워크를 학습시키는 **Learning Registration to Learn Segmentation (LRLS)** 방식이다. 

그러나 이러한 기존 방식들은 다음과 같은 치명적인 한계를 가진다:
1. **Registration 오류**: 영상 간의 강도(Intensity) 차이나 복잡한 변형으로 인해 정확한 정렬이 어려우며, 이는 곧바로 세그멘테이션 성능 저하로 이어진다.
2. **합성 데이터의 저품질**: LRLS 방식에서 생성된 합성 영상은 실제 영상의 정교한 해부학적 세부 구조를 충분히 반영하지 못하며, 이로 인해 모델의 일반화 성능이 떨어진다.

따라서 본 논문의 목표는 **실제 영상(Real Image)의 해부학적 특징을 직접적으로 학습에 활용**함으로써, 합성 데이터의 한계를 극복하고 정밀한 One-Shot 세그멘테이션을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **이미지 재구성(Image Reconstruction)으로 가이드되는 지식 증류(Knowledge Distillation)** 프레임워크를 도입하는 것이다. 

핵심 설계 직관은 다음과 같다. 합성 데이터만으로 학습하는 학생 네트워크(Student Network)는 실제 영상의 세부 구조를 파악하기 어렵다. 반면, 실제 영상의 재구성을 목표로 학습하는 교사 네트워크(Teacher Network)는 실제 영상의 해부학적 특징을 깊게 학습하게 된다. 이때 교사 네트워크가 추출한 고품질의 특징(Feature)을 학생 네트워크에게 전달(Distillation)함으로써, 학생 네트워크가 합성 데이터의 노이즈나 오류에 매몰되지 않고 실제 영상의 구조적 특성을 반영한 세그멘테이션을 수행하도록 유도하는 것이다.

## 📎 Related Works

논문에서는 One-Shot MIS의 기존 연구를 세 가지 범주로 나누어 설명한다.

1. **Atlas-Based Segmentation (ABS)**: ANTs와 같은 전통적인 도구나 VoxelMorph, CLMorph 같은 딥러닝 기반 Registration 모델을 사용하여 라벨을 전파한다. 하지만 영상 간 유사성에 지나치게 의존하며, 복잡한 변형이 발생하는 복부 CT 등의 데이터셋에서는 성능이 크게 떨어진다는 한계가 있다.
2. **Learning Registration to Learn Segmentation (LRLS)**: Registration 네트워크로 pseudo-dataset을 생성하고 이를 통해 세그멘테이션 모델을 학습시킨다. DataAug나 BRBS 같은 최신 방법론들이 이에 해당하지만, 여전히 실제 라벨링되지 않은 영상이 가진 풍부한 해부학적 정보를 충분히 활용하지 못하고 합성 이미지의 질에 의존하는 경향이 있다.
3. **Distillation Learning**: 복잡한 교사 모델의 지식을 단순한 학생 모델로 전이하는 기법이다. 기존 의료 영상 분야에서는 주로 모달리티 간 지식 전이에 사용되었으나, 본 논문은 이를 **재구성 태스크 $\rightarrow$ 세그멘테이션 태스크**라는 서로 다른 목적의 전이 학습으로 확장하여 적용했다.

## 🛠️ Methodology

전체 시스템은 크게 세 단계(데이터 증강 $\rightarrow$ 특징 증류 학습 $\rightarrow$ 추론)로 구성된다.

### 1. Registration 기반 데이터 증강
먼저, 단 하나의 Atlas $(x, l_x)$와 라벨이 없는 영상 집합 $Y=\{y_i\}$를 이용하여 합성 학습 데이터를 생성한다. CLMorph의 변형 모델을 사용하여 Atlas $x$를 대상 영상 $y_i$에 정렬시키는 변형 필드(Deformation Field) $\phi$를 학습한다.

- **합성 데이터 생성**: warping 연산 $\circ$를 통해 다음과 같이 합성 영상과 라벨을 생성한다.
  $$\hat{y}_i = x \circ \phi, \quad \hat{l}_{y_i} = l_x \circ \phi$$
- **Feature-Level Contrastive Learning**: Registration 성능을 높이기 위해 대조 학습(Contrastive Learning)을 도입한다. Atlas와 대상 영상 쌍의 유사도는 높이고, 다른 영상과의 유사도는 낮추는 $\mathcal{L}_{contrast}$를 통해 더욱 정교한 특징 표현을 학습한다.
  $$\mathcal{L}_{contrast}(H_u, H_a) = -\log \frac{\exp(\text{sim}(H_u, H_a)/\tau)}{\sum_{i \in N, i \neq y_i} \exp(\text{sim}(H_u, H_a)/\tau)}$$

### 2. 재구성을 통한 특징 증류 학습 (Feature Distillation Learning)
본 논문의 핵심 구조로, Teacher-Student 네트워크 아키텍처를 사용한다. 두 네트워크 모두 Residual Join U-Net 구조를 기반으로 한다.

- **Teacher Network (Rec Head)**: 실제 영상 $y_i$를 입력받아 이를 다시 재구성하는 과정을 통해 실제 영상의 해부학적 특징을 학습한다.
- **Student Network (Seg Head)**: 앞서 생성한 합성 영상 $\hat{y}_i$와 라벨 $\hat{l}_{y_i}$를 입력받아 세그멘테이션을 수행한다.
- **Hint Loss ($\mathcal{L}_{hint}$)**: 교사 네트워크의 특징 $\phi_{C_i}$와 학생 네트워크의 특징 $\phi_{M_i}$ 사이의 거리를 좁히기 위해 **코사인 유사도(Cosine Similarity)** 기반의 손실 함수를 사용한다.
  $$\mathcal{L}_{hint}(\phi_{C_i}, \phi_{M_i}) = \sum_{i=1}^{N} (1 - \cos(\phi_{C_i}, \phi_{M_i}))$$
  코사인 유사도는 단순한 $L_2$ 거리보다 각도 관계(Angular relationship)를 강조하므로, 해부학적 구조와 경계선을 학습하는 데 더 효과적이다.

### 3. 최적화 및 학습 절차
전체 학습 목표 함수는 다음과 같이 정의된다.
- **Registration 단계**: $\mathcal{L}_{reg} = \mathcal{L}_{sim} + \alpha \mathcal{L}_{smooth} + \beta \mathcal{L}_{contrast}$
- **Distillation 단계**: $\mathcal{L}_{kd} = \mathcal{L}_{seg} + \lambda_{recon} \mathcal{L}_{recon} + \lambda_{hint} \mathcal{L}_{hint}$
  - $\mathcal{L}_{seg}$: Cross-Entropy 기반 세그멘테이션 손실
  - $\mathcal{L}_{recon}$: 실제 영상과 재구성 영상 사이의 MSE 손실
  - $\mathcal{L}_{hint}$: 위에서 정의한 특징 전이 손실

최종 추론 단계에서는 무거운 교사 네트워크를 버리고, 경량화된 **학생 네트워크만을 사용**하여 새로운 영상에 대한 세그멘테이션을 수행한다.

## 📊 Results

### 실험 설정
- **데이터셋**: OASIS (뇌 MRI, 35개 구조), BCV (복부 CT, 간/비장/신장), VerSe (척추 CT, 13개 구조).
- **비교 대상**: Fully supervised (상한선), LS(U-Net, ResUNet), Trad(Rigid, Affine, SyN), ABS(VoxelMorph, TransMorph, CLMorph), LRLS(DataAug, BRBS).
- **평가 지표**: Dice Similarity Coefficient (DSC $\uparrow$), 95th percentile Hausdorff Distance ($HD_{95} \downarrow$).

### 주요 결과
- **정량적 성과**: 제안 방법론은 모든 데이터셋에서 SOTA One-Shot 방법론들보다 높은 DSC와 낮은 $HD_{95}$를 기록하였다.
  - **OASIS**: DSC $0.854$ (SOTA DataAug $0.823$ 대비 향상)
  - **BCV**: DSC $0.846$ (SOTA DataAug $0.822$ 대비 향상)
  - **VerSe**: DSC $0.924$ (SOTA BRBS $0.892$ 대비 향상)
- **정성적 성과**: 특히 뇌의 시상(Thalamus)이나 제3뇌실 같은 작은 구조, 그리고 복부 장기의 복잡한 외곽선에서 기존 방법론보다 훨씬 매끄럽고 정확한 경계를 생성함을 확인하였다.

### 절제 연구 (Ablation Study)
- **Hint Loss의 영향**: $L_2$ loss보다 코사인 유사도 loss를 사용했을 때 DSC가 유의미하게 상승하였다.
- **데이터의 영향**: 라벨 없는 데이터의 양이 증가함에 따라 성능이 빠르게 향상되었으며, 약 20%의 데이터만으로도 성능의 상당 부분이 회복됨을 보였다.
- **전이 계층**: 네트워크의 마지막 두 개 층(Layer)의 특징을 전이했을 때 가장 좋은 성능이 나타났다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 One-Shot 학습의 고질적인 문제인 '합성 데이터의 저품질' 문제를 **Task-specific Knowledge Distillation**으로 해결하였다. 교사 네트워크가 수행하는 '이미지 재구성'이라는 태스크는 입력 이미지의 픽셀 수준 정보를 보존해야 하므로, 자연스럽게 정교한 해부학적 특징 맵을 생성하게 된다. 학생 네트워크는 이 특징 맵을 가이드 삼아 학습함으로써, 합성 데이터만으로는 도달할 수 없었던 실제 영상의 세부 구조를 학습할 수 있게 된 것이다.

### 한계 및 비판적 논의
1. **데이터 이질성 문제**: 저자들도 언급했듯이 의료 영상의 높은 이질성(Heterogeneity)으로 인해, 매우 제한된 샘플만으로 학습한 모델이 완전히 새로운 도메인의 데이터에 대해서도 동일한 일반화 성능을 보일지는 미지수이다.
2. **학습 복잡도**: 추론 단계에서는 경량 모델을 사용하지만, 학습 단계에서는 Registration 네트워크와 Teacher-Student 네트워크를 모두 학습시켜야 하므로 초기 학습 비용과 메모리 소모가 크다.
3. **가정의 한계**: 본 연구는 Atlas가 테스트 셋과 어느 정도 유사하다는 가정 하에 최적의 Atlas를 선택하여 실험하였다. 만약 매우 이질적인 Atlas만 주어진 상황에서의 강건성(Robustness)에 대한 분석은 부족하다.

## 📌 TL;DR

이 논문은 단 하나의 라벨링 데이터만 사용하는 **One-Shot 의료 영상 세그멘테이션**에서, 합성 데이터의 품질 저하 문제를 해결하기 위해 **이미지 재구성 기반의 지식 증류(Distillation)** 프레임워크를 제안한다. 실제 영상을 재구성하는 교사 네트워크의 고품질 특징을 세그멘테이션을 수행하는 학생 네트워크에게 전이함으로써, MRI와 CT 등 다양한 모달리티에서 기존 SOTA 방법론을 뛰어넘는 정밀한 세그멘테이션 성능을 달성하였다. 이 연구는 라벨링 비용이 극심한 의료 AI 분야에서 실제 영상의 정보를 효율적으로 활용하는 새로운 학습 패러다임을 제시했다는 점에서 가치가 크다.