# Surgical-DeSAM: Decoupling SAM for Instrument Segmentation in Robotic Surgery

Yuyang Sheng, Sophia Bano, Matthew J. Clarkson, Mobarakol Islam (2024)

## 🧩 Problem to Solve

본 논문은 로봇 수술 환경에서 수술 도구의 인스턴스 분할(Instance Segmentation)을 자동화하는 문제를 다룬다. 최근 등장한 Segment Anything Model (SAM)은 포인트, 텍스트, 바운딩 박스와 같은 프롬프트를 통해 매우 뛰어난 분할 성능을 보여주었으나, 안전이 최우선인 수술 환경에 그대로 적용하기에는 다음과 같은 치명적인 한계가 존재한다.

첫째, 지도 학습을 위한 프레임별 프롬프트 데이터가 부족하며, 둘째, 실시간 추적 애플리케이션에서 매 프레임마다 사람이 수동으로 프롬프트를 제공하는 것은 불가능하며, 셋째, 오프라인 애플리케이션이라 하더라도 프롬프트를 일일이 주석 처리하는 비용이 지나치게 높다. 따라서 본 연구의 목표는 추가적인 수동 프롬프트 입력 없이도 실시간으로 수술 도구를 정밀하게 분할할 수 있는 자동화된 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM의 무거운 이미지 인코더를 제거하고, 이를 객체 탐지 모델인 DETR의 인코더로 대체하여 프롬프트 생성과 분할 과정을 하나로 통합하는 **Surgical-DeSAM** 구조를 설계한 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Swin-DETR 설계**: 기존 DETR의 ResNet50 백본을 Swin-transformer로 교체하여 수술 도구 탐지 성능을 향상시킨 Swin-DETR을 제안하였다.
2. **Decoupling SAM (DeSAM)**: SAM의 이미지 인코더를 DETR의 인코더 출력으로 대체함으로써, DETR이 생성한 바운딩 박스 프롬프트가 SAM의 마스크 디코더로 즉시 전달되는 구조를 구현하였다.
3. **End-to-End 학습**: 탐지(Detection)와 분할(Segmentation)을 동시에 학습시켜 프롬프트 생성부터 마스크 추출까지의 과정을 최적화하였다.

## 📎 Related Works

논문에서는 SAM과 DETR이라는 두 가지 핵심 아키텍처를 기반으로 한다.

- **SAM (Segment Anything Model)**: 거대한 데이터셋으로 학습된 파운데이션 모델로, 프롬프트 기반의 범용적 분할 능력을 갖추고 있다. 하지만 수술 도구 분할에 적용 시 인터랙티브한 프롬프트 입력이 필수적이라는 점이 실용성을 저해한다.
- **DETR (DEtection TRansformer)**: CNN 백본과 트랜스포머 인코더-디코더를 사용하여 객체 탐지를 수행하는 모델이다. 기존에는 주로 ResNet50을 백본으로 사용하였다.

기존의 수술 도구 분할 연구들은 주로 특정 데이터셋에 과적합된 세그멘테이션 모델을 사용하거나, SAM을 활용하더라도 수동 프롬프트에 의존하는 경향이 있었다. 본 논문은 탐지 모델을 통해 프롬프트를 자동 생성하고, SAM의 무거운 인코더를 제거하여 효율성을 높였다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

Surgical-DeSAM의 전체 흐름은 다음과 같다:
$\text{입력 이미지} \rightarrow \text{Swin-Transformer (특징 추출)} \rightarrow \text{DETR Encoder} \rightarrow \text{DETR Decoder (바운딩 박스 생성)} \rightarrow \text{SAM Prompt Encoder} \rightarrow \text{SAM Mask Decoder} \rightarrow \text{최종 세그멘테이션 마스크}$

### 주요 구성 요소 및 역할

1. **Swin-DETR**:
   - 기존 DETR의 ResNet50을 **Swin-transformer**로 교체하였다. Swin-transformer는 Shifted Window 기반의 계층적 구조를 통해 self-attention 계산 효율을 높이고 더 나은 특징 표현력을 제공한다.
   - ResNet50은 특징 맵 $f_{\text{resnet}} \in \mathbb{R}^{d \times H \times W}$를 $f \in \mathbb{R}^{d \times HW}$로 차원을 축소하는 과정이 필요하지만, Swin-transformer는 직접 $f_{\text{swin}} \in \mathbb{R}^{d \times HW}$ 형태의 출력을 생성하여 DETR 인코더에 효율적으로 전달한다.

2. **Decoupling SAM (DeSAM)**:
   - SAM의 무거운 이미지 인코더를 제거하고, 대신 DETR 인코더의 출력을 직접 SAM의 마스크 디코더에 입력으로 사용한다.
   - DETR 디코더에서 예측된 바운딩 박스는 SAM의 프롬프트 인코더를 통해 임베딩되어 마스크 디코더의 가이드 역할을 수행한다.

### 학습 절차 및 손실 함수

모델은 탐지 바운딩 박스의 정답(Ground-truth)과 분할 마스크의 정답을 모두 사용하여 End-to-End로 학습된다. 전체 손실 함수 $\text{Loss}_{\text{total}}$은 다음과 같이 정의된다.

$$\text{Loss}_{\text{total}} = L_{\text{box}} + L_{\text{dsc}}$$

- $L_{\text{box}}$: 탐지 작업을 위한 손실 함수로, $\text{GIoU}$ (Generalized Intersection over Union) 손실과 $l_1$ 손실을 결합하여 사용한다.
- $L_{\text{dsc}}$: 분할 작업을 위한 손실 함수로, $\text{Dice Similarity Coefficient (DSC)}$ 손실을 사용하여 예측 마스크와 정답 마스크 간의 유사도를 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MICCAI 수술 도구 분할 챌린지의 EndoVis 2017 및 EndoVis 2018 데이터셋을 사용하였다.
- **평가 지표**: $\text{mIoU}$ (mean Intersection over Union)와 $\text{DICE}$ score를 사용하여 성능을 측정하였다.
- **최적화**: AdamW 옵티마이저 (Learning rate $10^{-4}$, Weight decay $0.1$)를 사용하였다.

### 정량적 결과

Surgical-DeSAM은 기존 SOTA(State-of-the-art) 모델들과 비교하여 압도적인 성능 향상을 보였다.

- **EndoVis 2017**: $\text{DICE}$ score **89.62** 달성 (TraSeTR의 65.21, ISINet의 62.8 대비 매우 높음)
- **EndoVis 2018**: $\text{DICE}$ score **90.70** 달성 (TraSeTR의 81.10, ISINet의 78.30 대비 높음)

### Ablation Study (백본 비교)

Swin-transformer가 ResNet50보다 우수함을 증명하기 위해 비교 실험을 진행하였다.

- **탐지 작업**: Swin-transformer 기반 모델이 ResNet50 기반보다 $\text{mAP}$ 기준 약 $2.7\%$ 더 높은 성능을 보였다.
- **분할 작업**: Swin-transformer 기반 Surgical-DeSAM이 ResNet50 기반보다 $\text{DICE}$ score에서 **$7.1\%$ 더 높은 수치**를 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 SAM의 강력한 분할 능력을 유지하면서도, 실시간 시스템의 최대 걸림돌인 '수동 프롬프트' 문제를 DETR과의 통합을 통해 영리하게 해결하였다. 특히 SAM의 무거운 이미지 인코더를 제거하고 DETR의 인코더를 공유함으로써 연산 효율성을 높인 점이 돋보인다. 또한, 정밀한 바운딩 박스 탐지가 선행된 후 분할이 이루어지므로, 배경을 도구로 오인하는 False Positive 사례가 거의 발생하지 않는다는 정성적 이점을 확인하였다.

### 한계 및 논의사항

논문에서는 모델의 성능적 우수성을 입증하였으나, 실제 수술 환경에서의 완전한 신뢰성을 보장하기 위한 **강건성(Robustness)**과 **신뢰성(Reliability)**에 대한 추가적인 검증이 필요함을 언급하였다. 또한, 다양한 수술 도구 종류나 조명 변화, 혈흔으로 인한 가려짐(Occlusion) 상황에서의 성능 변화에 대한 심층적인 분석은 본문에서 명시적으로 다루어지지 않았다.

## 📌 TL;DR

본 연구는 SAM의 프롬프트 의존성 문제를 해결하기 위해, Swin-transformer 기반의 DETR 탐지기와 SAM의 마스크 디코더를 결합한 **Surgical-DeSAM**을 제안하였다. SAM의 이미지 인코더를 DETR 인코더로 대체하는 'Decoupling' 전략을 통해 프롬프트 생성을 자동화하였으며, 그 결과 EndoVis 2017/2018 데이터셋에서 $\text{DICE}$ score 89.62/90.70라는 SOTA 성능을 달성하였다. 이 연구는 실시간 로봇 수술 보조 시스템에서 수동 입력 없는 고정밀 도구 분할을 가능하게 함으로써 향후 지능형 수술 로봇의 자율성 향상에 기여할 가능성이 크다.
