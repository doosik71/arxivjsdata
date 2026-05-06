# Semantic Image Segmentation: Two Decades of Research

Gabriela Csurka, Riccardo Volpi, and Boris Chidlovskii (2023)

## 🧩 Problem to Solve

본 논문은 지난 20년간의 Semantic Image Segmentation (SiS) 연구 흐름과 최근 5년간 급격히 성장한 Domain Adaptation for Semantic Image Segmentation (DASiS) 분야를 체계적으로 정리하는 것을 목표로 하는 종합 서베이 논문이다.

SiS는 이미지의 각 픽셀에 해당하는 semantic class를 라벨링하여 이미지의 전반적인 이해와 객체 간의 공간적 관계를 파악하는 핵심적인 컴퓨터 비전 작업이다. 이는 자율 주행 자동차(AD), 지능형 로봇, 의료 영상 분석 등 다양한 실세계 응용 분야에서 필수적이다. 그러나 SiS의 가장 큰 병목 현상은 픽셀 단위의 정밀한 어노테이션(pixel-wise annotation)에 막대한 비용과 시간이 소요된다는 점이다.

이를 해결하기 위해 그래픽 엔진을 이용한 합성 데이터(synthetic data) 생성이 대안으로 제시되었으나, 합성 이미지와 실제 이미지 사이의 Domain Shift로 인해 성능 저하가 발생한다. 따라서 본 논문은 SiS의 전반적인 방법론과 더불어, 합성 데이터로 학습된 모델을 실제 환경에 적응시키기 위한 Unsupervised Domain Adaptation (UDA) 및 DASiS 기법들을 상세히 분석하여 연구자들에게 포괄적인 가이드를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 주요 기여는 다음과 같다.

1. **SiS 방법론의 역사적 및 현대적 정리**: 초기 역사적 방법론부터 최신 Deep Learning 기반 모델, 그리고 최신 트렌드인 Transformer 기반 아키텍처까지 20년 치의 연구를 체계적으로 분류하고 분석하였다.
2. **DASiS의 상세 분석 프레임워크 제시**: Domain Adaptation 기법을 이미지 수준(Image-level), 특징 수준(Feature-level), 출력 수준(Output-level)의 세 가지 정렬(alignment) 관점에서 분류하고, 각 방법론의 특성을 정리한 상세 표(Table 2.1)를 제공하였다.
3. **보조 학습 기법 및 확장 시나리오 탐색**: Pseudo-labeling, Entropy minimization, Curriculum learning 등 성능 향상을 위한 보조 기법들과 Multi-source, Source-free, Domain Generalization 등 복잡한 실제 환경을 가정한 다양한 DA 시나리오를 분석하였다.
4. **종합적인 벤치마크 및 데이터셋 가이드 제공**: SiS와 DASiS에서 널리 사용되는 데이터셋, 평가 지표(mIoU 등), 그리고 효율성과 정확도 간의 Trade-off 및 모델의 취약성(Vulnerability)에 대한 논의를 포함하였다.

## 📎 Related Works

논문은 기존의 SiS 및 DA 서베이 연구들과의 차별점을 다음과 같이 제시한다.

- **SiS 서베이 측면**: 기존 연구(Minaee et al., 2020 등)들이 초기 딥러닝 모델에 집중했다면, 본 논문은 최신 Attention mechanism과 Transformer 기반 모델들을 포함하여 분석 범위를 확장하였다.
- **DA 서베이 측면**: 기존의 DA 서베이들이 주로 이미지 분류(Image Classification) 작업에 치중되어 SiS에 대한 비중이 낮았던 반면, 본 논문은 SiS 특유의 공간적 구조와 픽 uma-pixel 수준의 복잡성을 고려한 DASiS 전용 기법들을 집중적으로 다루었다.
- **포괄성**: 단순한 방법론 나열을 넘어, 데이터셋-평가 지표-최신 트렌드-관련 작업(Instance/Panoptic Segmentation)을 모두 아우르는 책 형태의 포괄적인 레퍼런스를 지향한다.

## 🛠️ Methodology

본 논문은 서베이 논문으로서 특정 알고리즘을 제안하기보다, 기존의 방대한 방법론을 체계적인 분류 체계(Taxonomy)에 따라 설명한다.

### 1. Semantic Image Segmentation (SiS) 파이프라인

SiS의 발전 과정을 세 단계로 구분하여 설명한다.

- **역사적 방법론**: 지역적 외형(Local appearance) 모델링 $\rightarrow$ 지역적/전역적 일관성(Local/Global consistency) 강화(예: CRF) $\rightarrow$ 사전 지식(Prior knowledge) 활용 순으로 발전하였다.
- **딥러닝 기반 모델**:
  - **FCN**: Fully Convolutional Network를 통해 임의 크기의 입력에서 밀집 예측(dense prediction)을 수행한다.
  - **Encoder-Decoder**: U-Net, SegNet과 같이 인코더에서 압축된 특징을 디코더에서 복원하며, Skip connection을 통해 공간 정보를 보존한다.
  - **Pyramidal/Dilated Conv**: PSPNet, DeepLab 등은 Atrous Convolution과 Pyramid Pooling을 통해 수용 영역(Receptive field)을 확장하여 전역 문맥(Global context)을 파악한다.
  - **Transformer**: ViT, Swin Transformer 등을 도입하여 CNN의 지역적 제한을 넘어 전역적인 Self-attention을 수행한다.
- **손실 함수**: 가장 기본이 되는 Pixel-wise Cross-Entropy Loss는 다음과 같이 정의된다.
    $$L_{ce} = -\mathbb{E}_{(X,Y)} \left[ \sum_{h,w} y^{(h,w)} \cdot \log(p(F(x^{(h,w)}))) \right]$$
    여기서 $y$는 ground-truth one-hot 벡터이며, $p$는 모델 $F$가 예측한 클래스 확률 벡터이다. 클래스 불균형 문제를 해결하기 위해 Weighted Cross-Entropy나 IoU 직접 최적화 방식이 사용된다.

### 2. Domain Adaptation for SiS (DASiS) 프레임워크

UDA의 목표는 소스 도메인 $P_S(X, Y)$에서 학습한 모델을 라벨이 없는 타겟 도메인 $P_T(X)$에 적응시키는 것이다.

- **Feature-level Adaptation**: 소스와 타겟의 특징 분포 간의 거리(예: Maximum Mean Discrepancy, MMD)를 최소화하거나, Domain Discriminator를 이용한 적대적 학습(Adversarial Training)을 통해 도메인 불변 특징(Domain-invariant features)을 학습한다.
- **Image-level Adaptation**: Style Transfer(예: CycleGAN)를 통해 소스 이미지의 외형을 타겟 도메인처럼 변환하여 학습한다. 이때 구조적 보존을 위해 Cycle consistency loss $\text{L}_{\text{cycle}}$와 세만틱 일관성 손실 $\text{L}_{\text{SemCons}}$를 사용한다.
- **Output-level Adaptation**: 특징 공간이 아닌, 최종 예측 결과인 클래스 확률 맵(likelihood map) 수준에서 적대적 정렬을 수행하여 도메인 간 간극을 줄인다.

### 3. 보조 및 확장 기법

- **Self-training**: 타겟 데이터에 대해 모델이 예측한 값 중 신뢰도가 높은 것을 Pseudo-label로 사용하여 반복 학습한다.
- **Entropy Minimization (TEM)**: 타겟 예측의 엔트로피를 낮춤으로써 모델이 타겟 도메인에서 더 확신 있는(confident) 예측을 하도록 유도한다.
- **Advanced Scenarios**: Source-free DA(소스 데이터 없이 모델만 이용), Domain Generalization(타겟 데이터를 전혀 모르는 상태에서 일반화), Online Adaptation(실시간 데이터 스트림에 적응) 등을 다룬다.

## 📊 Results

본 논문은 직접적인 실험 결과보다는 기존 연구들이 사용한 벤치마크와 지표를 정리하여 제시한다.

### 1. 주요 데이터셋 및 벤치마크

- **Object Segmentation**: PASCAL VOC, MS COCO.
- **Image Parsing**: ADE20K.
- **Autonomous Driving (AD)**: Cityscapes(실제), GTA-5 및 SYNTHIA(합성).
- **DASiS 벤치마크**: 가장 대표적인 설정은 $\text{GTA-5} \rightarrow \text{Cityscapes}$ (Sim-to-Real) 적응 작업이다.

### 2. 평가 지표

- **Mean IoU (mIoU)**: 각 클래스별 Intersection over Union의 평균으로, SiS의 표준 지표이다.
    $$\text{mIoU} = \frac{1}{C} \sum_{i} \frac{n_{ii}}{t_i + \sum_{j} n_{ji} - n_{ii}}$$
- **Pixel Accuracy**: 전체 픽셀 중 정답을 맞춘 비율이다.
- **Boundary-aware Metrics**: Trimap accuracy나 BCM score를 통해 경계선 검출 능력을 평가한다.

### 3. 분석 결과의 중요성

논문은 단순히 정확도(mIoU)만 높이는 것이 아니라, 실시간성(Efficiency)과 강건성(Robustness)의 조화가 중요함을 강조한다. 특히, 딥러닝 모델이 인간이 보기엔 무해한 미세한 섭동(Adversarial perturbations)에 취약하며, 학습 데이터에 없는 Unseen classes가 나타났을 때의 대응 능력이 실세계 적용의 핵심임을 지적한다.

## 🧠 Insights & Discussion

### 강점 및 통찰

- 본 보고서는 SiS의 전체 역사를 훑으며, 단순한 모델 구조의 변화가 아니라 "어떻게 하면 효율적으로 라벨링 비용을 줄이고(Synthetic data $\rightarrow$ DA), 어떻게 하면 전역적 문맥을 더 잘 파악할 것인가(CNN $\rightarrow$ Transformer)"라는 핵심 문제 해결 과정을 잘 보여준다.
- 특히 DASiS를 세 가지 수준(Image, Feature, Output)으로 계층화하여 분석한 점은 향후 새로운 DA 기법을 설계할 때 매우 유용한 가이드라인이 된다.

### 한계 및 논의사항

- **데이터셋 의존성**: 많은 DASiS 연구가 $\text{GTA-5} \rightarrow \text{Cityscapes}$ 설정에 과하게 의존하고 있어, 실제 도로의 다양한 변수(날씨, 조명, 지리적 특성)를 모두 반영하기에는 한계가 있다.
- **평가 프로토콜의 모호성**: UDA 모델의 하이퍼파라미터 튜닝 시 타겟 도메인의 라벨을 일부 사용하는 경우가 많은데, 이는 UDA의 기본 전제(타겟 라벨 없음)와 충돌하는 지점이다. 이에 대한 더 엄격한 평가 프로토콜이 필요하다.
- **미래 방향**: 최근의 Foundation Models(예: CLIP, SAM)가 제공하는 강력한 제로샷(Zero-shot) 전이 능력을 DASiS와 어떻게 결합할 것인지가 차세대 연구의 핵심이 될 것이다.

## 📌 TL;DR

본 논문은 지난 20년간의 **Semantic Image Segmentation (SiS)** 연구와 최근의 **Domain Adaptation (DASiS)** 기법을 집대성한 종합 서베이이다. 딥러닝의 발전으로 성능은 비약적으로 향상되었으나, **픽셀 단위 라벨링 비용**이라는 근본적 문제가 여전히 존재하며, 이를 극복하기 위해 **합성 데이터 $\rightarrow$ 실제 데이터로의 도메인 적응**이 핵심 연구 분야임을 강조한다. 특히 DASiS를 이미지/특징/출력 수준의 정렬로 체계화하여 분석함으로써, 실세계의 자율 주행 및 의료 영상 시스템 구축을 위한 이론적 기반과 실무적 가이드를 제공한다.
