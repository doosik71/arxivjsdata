# Surgical-DeSAM: Decoupling SAM for Instrument Segmentation in Robotic Surgery

Yuyang Sheng, Sophia Bano, Matthew J. Clarkson, Mobarakol Islam (2024)

## 🧩 Problem to Solve

본 논문은 로봇 수술 환경에서 수술 도구의 인스턴스 분할(Instance Segmentation)을 자동화하는 문제를 다룬다. 최근 등장한 Segment Anything Model (SAM)은 포인트, 텍스트, Bounding Box와 같은 프롬프트(Prompt)를 통해 뛰어난 분할 성능을 보여주었으나, 안전이 최우선인 수술 환경에서는 다음과 같은 이유로 직접적인 프롬프트 입력 방식의 적용이 불가능하다.

첫째, 지도 학습을 위한 프레임별 프롬프트 데이터가 부족하다. 둘째, 실시간 추적(Real-time tracking) 애플리케이션에서 매 프레임마다 사람이 프롬프트를 입력하는 것은 비현실적이다. 셋째, 오프라인 애플리케이션을 위해 모든 프레임에 프롬프트를 어노테이션 하는 비용이 매우 높다. 따라서 본 연구의 목표는 추가적인 수동 프롬프트 입력 없이도 실시간으로 수술 도구를 정밀하게 분할할 수 있는 자동화된 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM의 무거운 Image Encoder를 제거하고, 이를 객체 탐지 모델인 DETR 계열의 인코더로 대체하여 '디커플링(Decoupling)'하는 것이다. 주요 기여 사항은 다음과 같다.

- **Surgical-DeSAM 제안**: Bounding Box 프롬프트를 자동으로 생성하여 SAM에 전달함으로써 프롬프트 입력 과정을 자동화한 아키텍처를 제안한다.
- **Swin-DETR 설계**: DETR의 기존 ResNet50 백본을 Swin-Transformer로 교체하여 수술 도구 탐지 성능과 특징 표현(Feature Representation) 능력을 향상시켰다.
- **Decoupling SAM (DeSAM)**: SAM의 Image Encoder를 제거하고 DETR의 Encoder 출력을 직접 Mask Decoder에 입력하는 구조를 설계하여, 탐지와 분할이 통합된 엔드 투 엔드(End-to-End) 학습을 가능하게 하였다.

## 📎 Related Works

논문에서는 SAM과 DETR 두 가지 핵심 모델을 언급한다. SAM은 10억 개 이상의 고품질 마스크로 학습된 파운데이션 모델로, Image Encoder, Prompt Encoder, Mask Decoder로 구성된다. 하지만 SAM은 객체 레이블을 스스로 생성할 수 없으며 인터랙티브한 프롬프트가 필수적이라는 한계가 있다.

DETR (DEtection TRansformer)은 CNN 백본과 트랜스포머 인코더-디코더를 사용하여 객체 탐지를 수행하는 모델이다. 기존의 수술 도구 분할 연구들은 주로 시맨틱 분할이나 특정 아키텍처에 의존했으나, 본 논문은 이러한 최신 탐지 모델(DETR)과 분할 모델(SAM)의 장점을 결합하여 프롬프트 의존성 문제를 해결함으로써 기존 방식과 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

Surgical-DeSAM은 크게 **Swin-DETR**을 통한 도구 탐지와 **Decoupling SAM**을 통한 마스크 생성의 두 단계로 구성된다. Swin-DETR이 이미지에서 수술 도구의 Bounding Box를 예측하면, 이 정보가 SAM의 Prompt Encoder로 전달되고, DETR Encoder에서 추출된 특징 맵이 SAM의 Mask Decoder로 전달되어 최종 분할 결과물을 생성한다.

### 주요 구성 요소 및 역할

1. **Swin-DETR**:
   - **Swin-Transformer**: 기존 ResNet50 대신 사용되어 계층적 구조와 Shifted Window 기반의 self-attention을 통해 효율적인 이미지 특징을 추출한다.
   - **DETR Encoder & Decoder**: 추출된 특징을 바탕으로 객체 쿼리를 생성하고, 최종적으로 도구의 클래스와 Bounding Box 좌표를 예측한다.
2. **Decoupling SAM (DeSAM)**:
   - SAM의 무거운 Image Encoder를 제거하고, DETR Encoder의 출력을 직접 Mask Decoder의 입력으로 사용한다.
   - **Prompt Encoder**: Swin-DETR이 예측한 Bounding Box를 임베딩 벡터로 변환한다.
   - **Mask Decoder**: DETR Encoder의 특징 맵과 Prompt Encoder의 임베딩을 결합하여 정밀한 세그멘테이션 마스크를 생성한다.

### 학습 절차 및 손실 함수

모델은 탐지(Detection)와 분할(Segmentation) 작업을 동시에 수행하도록 엔드 투 엔드(End-to-End)로 학습된다. 전체 손실 함수 $\text{Loss}_{\text{total}}$은 다음과 같이 정의된다.

$$\text{Loss}_{\text{total}} = L_{\text{box}} + L_{\text{dsc}}$$

여기서 각 항의 의미는 다음과 같다.

- $L_{\text{box}}$: 객체 탐지 성능을 높이기 위한 손실 함수로, GIoU (Generalized Intersection over Union) 손실과 $l_1$ 손실을 결합하여 사용한다.
- $L_{\text{dsc}}$: 분할 정확도를 측정하는 Dice Coefficient Similarity (DSC) 손실 함수이다.

## 📊 Results

### 실험 설정

- **데이터셋**: MICCAI 수술 도구 분할 챌린지 데이터셋인 EndoVis 2017 및 EndoVis 2018을 사용하였다.
- **지표**: mIoU (mean Intersection over Union)와 DICE score를 사용하여 성능을 평가하였다.
- **비교 대상**: TernausNet, MF-TAPNet, Dual-MF, TrackFormer, ISINet, TraSeTR, S3Net, SurgicalSAM 등 최신 SOTA 모델들과 비교하였다.

### 정량적 결과

실험 결과, Surgical-DeSAM은 두 데이터셋 모두에서 다른 SOTA 모델들을 크게 상회하는 성능을 보였다.

- **EndoVis 2017**: DICE score $89.62\%$, mIoU $82.41\%$ 달성.
- **EndoVis 2018**: DICE score $90.70\%$, mIoU $84.91\%$ 달성.

### 정성적 결과 및 분석

Swin-DETR이 정확한 Bounding Box를 예측하기 때문에, 이를 기반으로 한 분할 결과에서 False Positive(오탐지)가 거의 발생하지 않는 것으로 확인되었다. 또한, Ablation Study를 통해 ResNet50보다 Swin-Transformer를 백본으로 사용했을 때 탐지 작업에서 mAP가 $2.7\%$ 향상되었으며, 분할 작업에서 DICE score가 $7.1\%$ 향상됨을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 SAM의 강력한 분할 능력을 유지하면서도, 수술 환경의 실시간 제약 조건을 해결하기 위해 '프롬프트 생성의 자동화'와 '모델 구조의 경량화(디커플링)'라는 전략을 적절히 사용하였다. 특히 SAM의 Image Encoder를 제거하고 DETR의 인코더를 공유함으로써 계산 효율성을 높이고 탐지와 분할 사이의 정렬(Alignment)을 최적화한 점이 돋보인다.

다만, 논문에서는 주로 공개 데이터셋에서의 성능 향상에 집중하고 있으며, 실제 수술실의 다양한 조명 변화나 도구의 가려짐(Occlusion) 상황에서의 강건성(Robustness)에 대한 상세한 분석은 부족한 편이다. 또한, 실시간 성능을 주장하고 있으나 구체적인 추론 속도(FPS)에 대한 수치는 명시되지 않았다.

## 📌 TL;DR

본 연구는 SAM의 인터랙티브 프롬프트 입력 문제를 해결하기 위해, Swin-Transformer 기반의 DETR로 Bounding Box를 자동 생성하고 SAM의 Image Encoder를 제거하여 결합한 **Surgical-DeSAM**을 제안한다. 이 모델은 EndoVis 2017/2018 데이터셋에서 DICE score 약 $90\%$ 수준의 높은 성능을 기록하며 SOTA를 달성하였다. 이 연구는 수동 입력 없이도 정밀한 수술 도구 분할이 가능함을 보여주어, 향후 로봇 수술의 자동화 및 실시간 가이드 시스템 구축에 중요한 기여를 할 것으로 보인다.
