# Embracing Imperfect Datasets: A Review of Deep Learning Solutions for Medical Image Segmentation

Nima Tajbakhsh, Laura Jeyaseelan, Qian Li, Jeffrey N. Chiang, Zhihao Wu, and Xiaowei Ding (2020)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 Convolutional Neural Networks(CNN) 기반의 모델들은 비약적인 성능 향상을 이루었으나, 이러한 모델들이 최적의 성능을 내기 위해서는 대규모의 대표성 있고 고품질인 어노테이션 데이터셋(Annotated Datasets)이 필수적이다. 하지만 의료 영상 분야의 특성상 데이터 획득 비용이 높고, 특히 전문의에 의한 정밀한 어노테이션 작업은 매우 비용이 많이 들기 때문에 완벽한 데이터셋을 구축하는 것은 현실적으로 매우 어렵다.

본 논문이 해결하고자 하는 핵심 문제는 '불완전한 데이터셋(Imperfect Datasets)' 상황에서도 효과적으로 작동하는 딥러닝 솔루션을 체계화하는 것이다. 여기서 불완전함은 크게 두 가지 방향으로 정의된다. 첫째는 학습에 사용할 수 있는 정답 데이터가 매우 적은 **Scarce Annotations(희소 어노테이션)** 문제이며, 둘째는 데이터는 존재하지만 정답의 질이 낮거나 불완전한 **Weak Annotations(약한 어노테이션)** 문제이다. 본 연구의 목표는 이러한 데이터 제약 조건을 극복하기 위한 최신 딥러닝 기법들을 리뷰하고, 각 방법론의 기술적 novelty와 경험적 결과를 분석하여 실무적인 권장 솔루션을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 의료 영상 분할에서 발생하는 데이터셋의 불완전성을 **Scarce Annotations**와 **Weak Annotations**라는 두 가지 큰 축으로 분류하고, 이를 해결하기 위한 방대한 솔루션들을 체계적인 텍스트와 표(Taxonomy) 형태로 정리했다는 점이다.

단순한 나열식 리뷰를 넘어, 각 방법론이 요구하는 데이터 자원(Data Requirements), 구현 난이도, 그리고 성능 이득(Performance Gain) 사이의 **비용-이득 트레이드오프(Cost-Gain Trade-off)** 관점에서 분석을 수행하였다. 이를 통해 연구자와 실무자가 자신이 처한 데이터 상황(예: 레이블이 전혀 없는 경우, 이미지 수준의 레이블만 있는 경우 등)에 따라 어떤 전략을 선택해야 하는지에 대한 가이드라인을 제시하였다.

## 📎 Related Works

기존의 의료 영상 관련 서베이 논문들은 다음과 같은 한계가 있었다.
- Litjens et al. (2017)은 의료 영상 전반의 딥러닝 솔루션을 다루었으나, 특정 문제(데이터 불완전성)에 집중하지 않았다.
- Yi et al. (2018)은 GAN의 활용에만 초점을 맞췄으며, Cheplygina et al. (2019)은 준지도 학습 및 전이 학습을 다루었으나 일반적인 의료 영상 분석 범위에 머물렀다.
- Zhang et al. (2019b)은 소규모 샘플 문제(Small Sample Size)만을 다루었고, Karimi et al. (2019)은 레이블 노이즈(Label Noise) 문제에 국한되었다.

본 논문은 특히 **'분할(Segmentation)'** 작업이 분류(Classification)나 탐지(Detection)보다 훨씬 강력한 지도 학습(Strong Supervision)을 필요로 하며, 따라서 데이터의 양과 질에 가장 민감하다는 점에 착안하여, 분할 작업에 특화된 데이터 부족 및 약한 지도 학습 솔루션을 통합적으로 분석했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

본 논문은 불완전한 데이터셋 문제를 두 가지 시나리오로 나누어 분석한다.

### 1. Scarce Annotations (희소 어노테이션) 해결 방안
데이터 양이 절대적으로 부족한 경우, 크게 세 가지 전략을 사용한다.

**가. 학습 데이터셋 확장 (Enlarging the Training Set)**
- **Data Augmentation**: 전통적인 공간/강도 변환 외에도, 두 이미지를 선형 결합하는 Mixup을 사용한다. Mixup의 수식은 다음과 같다.
  $$\tilde{x} = \lambda x_i + (1-\lambda)x_j$$
  $$\tilde{y} = \lambda y_i + (1-\lambda)y_j$$
  여기서 $\lambda$는 베타 분포에서 샘플링된 값이다. 또한 GAN을 이용한 합성 데이터 생성(Synthetic Augmentation)을 통해 도메인 내/외부에서 데이터를 생성한다.
- **External Labeled Data**: 자연 영상에서 학습된 모델을 가져오는 Transfer Learning, 도메인 간 간극을 줄이는 Domain Adaptation, 여러 데이터셋을 통합하는 Dataset Fusion을 활용한다.
- **Cost Effective Annotation**: 어떤 데이터를 우선적으로 레이블링할지 결정하는 Active Learning과, 전문가의 수정을 빠르게 반영하는 Interactive Segmentation을 통해 비용을 최적화한다.
- **Unlabeled Data**: 레이블이 없는 데이터를 활용하는 Self-supervised pre-training과 Semi-supervised learning을 적용한다.

**나. 학습 정규화 강화 (Strengthening Regularization)**
- **Altered Image Representation**: 3D 영상을 2.5D나 멀티뷰 뷰(Multi-view)로 변환하여 모델이 학습해야 할 복잡도를 낮춘다.
- **Multi-task Learning**: 분할과 함께 분류나 재구성(Reconstruction) 작업을 동시에 학습시켜 공유 인코더가 더 일반적인 특징을 학습하도록 유도한다.
- **Shape Regularization**: 결과물이 해부학적으로 타당하도록 제약을 가한다. 예를 들어, Star shape prior는 중심점과 내부 점 사이의 모든 경로가 내부여야 한다는 제약을 주어 구멍이 없는 매끄러운 마스크를 생성하게 한다.

**다. 사후 정제 (Post-segmentation Refinement)**
- **Conditional Random Fields (CRF)**: 픽셀 간의 관계를 모델링하여 경계를 날카롭게 다듬는다. 에너지 함수 $E(x|I)$는 다음과 같이 정의된다.
  $$E(x|I) = \sum_{i} \phi_u(x_i) + \sum_{i \neq j} \phi_p(x_i, x_j)$$
  여기서 $\phi_u$는 유니러리 포텐셜(Unary potential), $\phi_p$는 페어와이즈 포텐셜(Pairwise potential)이다. 최근에는 이를 RNN 형태로 구현한 RNN-CRF를 통해 end-to-end 학습을 수행한다.

### 2. Weak Annotations (약한 어노테이션) 해결 방안
정답의 질이 낮거나 불완전한 경우, 어노테이션의 유형에 따라 대응한다.

- **Image-level Labels**: 이미지 전체에 대한 레이블만 있는 경우, Class Activation Maps (CAM)를 통해 활성화 맵을 생성하거나, 이미지 전체를 하나의 '백(Bag)'으로 보는 Multiple Instance Learning (MIL)을 통해 픽셀 수준의 예측을 수행한다.
- **Sparse Labels**: 일부 슬라이스나 픽셀만 레이블링된 경우, 레이블이 있는 픽셀에 대해서만 손실 함수를 계산하는 Selective Loss를 사용하거나, Active Learning 등을 통해 빈 곳을 채우는 Mask Completion 기법을 사용한다.
- **Noisy Labels**: 경계가 부정확한 경우, 레이블의 신뢰도를 측정하여 가중치를 조절하는 Robust Loss를 사용하거나, 반복적인 정제 과정을 통해 노이즈를 제거한다.

## 📊 Results

본 논문은 개별 제안 방법론들의 실험 결과를 종합하여 분석하였다.

- **정량적 성과**: 
    - **Self-supervised learning**의 경우, Models Genesis와 같은 프레임워크가 3D 폐 결절 및 간 분할에서 IoU를 3~5포인트 향상시켰음을 확인하였다.
    - **Domain Adaptation**은 타겟 도메인의 레이블이 없을 때도 합성 데이터를 통해 실측 데이터 기반 모델과 유사한 수준의 성능에 도달할 수 있음을 보여주었다.
    - **Active Learning**은 전체 데이터의 20~50%만 레이블링하고도 전체 데이터를 사용했을 때와 유사한 성능을 낼 수 있음을 입증하였다.
- **방법론 비교**: 
    - 2D 분할에서는 CRF 기반의 사후 정제가 매우 효과적이지만, 3D 분할에서는 계산 복잡도와 하이퍼파라미터 튜닝의 어려움으로 인해 그 효과가 상대적으로 낮게 나타났다.
    - 전이 학습(Transfer Learning)은 2D 모델에서는 유효하나, 3D 의료 영상 모델로의 직접적인 적용은 여전히 도전적인 과제로 남아 있다.

## 🧠 Insights & Discussion

### 강점 및 추천 전략
저자들은 데이터 자원 수준에 따라 다음과 같은 전략을 추천한다.
1. **저자원 상황 (추가 데이터 없음)**: 전통적인 Data Augmentation과 더불어 Mixup, Shape Regularization, 그리고 2D의 경우 CRF 정제를 우선적으로 사용해야 한다.
2. **중자원 상황 (미레이블 데이터 존재)**: Self-supervised pre-training이 가장 유망하며, 구현이 쉽고 성능 향상 폭이 크다.
3. **고자원 상황 (전문가 투입 가능)**: Active Learning과 Interactive Segmentation을 결합하여 레이블링 비용을 최소화하면서 데이터셋을 확장하는 것이 가장 확실한 방법이다.

### 한계 및 비판적 해석
- **결합 전략의 부재**: 대부분의 연구가 하나의 제약 조건(예: Scarce 또는 Weak)만 해결하려 한다. 하지만 실제 임상 데이터는 두 가지 문제가 동시에 나타나는 경우가 많으므로, 이를 통합적으로 해결하는 파이프라인 연구가 필요하다.
- **3D 확장성의 한계**: 많은 기법들이 2D에서 성공했으나 3D로 넘어올 때 성능 저하가 발생하거나 계산 비용이 기하급수적으로 증가하는 경향이 있다. 이는 단순히 차원을 늘리는 것이 아니라 3D의 공간적 특성을 반영한 새로운 아키텍처가 필요함을 시사한다.
- **현실적 비용 측정**: Active Learning 연구들이 '레이블 수'의 감소는 강조하지만, 실제 전문가가 느끼는 '인지적 노력'이나 '시간 비용'에 대한 정밀한 분석은 부족한 편이다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 치명적인 문제인 **데이터 부족(Scarce)**과 **낮은 데이터 품질(Weak)** 문제를 해결하기 위한 딥러닝 기법들을 총망라한 리뷰 논문이다. 데이터 증강, 전이 학습, 준지도 학습, 정규화, 그리고 약한 지도 학습(CAM, MIL 등)을 체계적으로 분류하였으며, 특히 **데이터 자원-구현 난이도-성능** 사이의 트레이드오프를 분석하여 실무적인 선택 가이드를 제공하였다. 이 연구는 향후 불완전한 의료 데이터셋을 활용한 강건한(Robust) 분할 모델 설계의 이론적 토대가 될 것으로 기대된다.