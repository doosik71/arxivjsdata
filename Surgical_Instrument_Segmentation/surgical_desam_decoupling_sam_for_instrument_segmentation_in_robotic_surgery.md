# Surgical-DeSAM: Decoupling SAM for Instrument Segmentation in Robotic Surgery

Yuyang Sheng, Sophia Bano, Matthew J. Clarkson, Mobarakol Islam (2024)

## 🧩 Problem to Solve

본 논문은 로봇 수술 환경에서 수술 도구의 인스턴스 분할(Instance Segmentation)을 자동화하는 문제를 해결하고자 한다. 최근 Segment Anything Model (SAM)은 포인트, 텍스트, 바운딩 박스 등의 프롬프트(Prompt)가 주어졌을 때 매우 뛰어난 분할 성능을 보여주었다. 그러나 실제 수술과 같은 안전 필수적(safety-critical) 상황에서는 다음과 같은 이유로 SAM의 프롬프트 기반 방식이 적용되기 어렵다.

첫째, 지도 학습을 위한 프레임별 프롬프트 데이터가 부족하다. 둘째, 실시간 트래킹 애플리케이션에서 매 프레임마다 수동으로 프롬프트를 제공하는 것은 비현실적이다. 셋째, 오프라인 분석을 위한 애플리케이션에서도 프롬프트를 일일이 주석(Annotation)하는 작업의 비용이 매우 높다. 따라서 본 연구의 목표는 외부의 수동 프롬프트 없이도 수술 도구를 실시간으로 정밀하게 분할할 수 있는 자동화된 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM의 무거운 이미지 인코더를 제거하고, 이를 객체 탐지 모델인 DETR의 인코더로 대체하여 프롬프트 생성과 분할 과정을 하나의 파이프라인으로 통합하는 'Decoupling SAM' 구조를 제안하는 것이다.

주요 기여 사항은 다음과 같다.

1. **Swin-DETR 설계**: DETR의 기존 ResNet50 백본을 Swin-Transformer로 교체하여 수술 도구 탐지 성능을 향상시켰다.
2. **Surgical-DeSAM 제안**: SAM의 이미지 인코더를 DETR의 인코더 출력으로 대체하여, 탐지된 바운딩 박스가 자동으로 SAM의 프롬프트로 입력되는 디커플링 구조를 구현하였다.
3. **End-to-End 학습**: 탐지(Detection)와 분할(Segmentation) 작업을 동시에 학습시켜 실시간으로 프롬프트 생성부터 마스크 추출까지 가능하게 하였다.

## 📎 Related Works

본 논문은 크게 SAM과 DETR라는 두 가지 기반 모델을 언급한다.

- **SAM (Segment Anything Model)**: 방대한 데이터셋으로 학습된 파운데이션 모델로, 프롬프트 기반 분할에 매우 강력하다. 하지만 수술 영상과 같은 실시간 도메인에서는 매 프레임 프롬프트를 입력해야 한다는 치명적인 한계가 있으며, 객체 라벨링(Object label segmentation) 기능이 부족하다.
- **DETR (DEtection TRansformer)**: Transformer 구조를 사용하여 객체 탐지를 수행하는 모델로, CNN 백본과 Transformer 인코더-디코더를 통해 바운딩 박스를 예측한다.

기존의 수술 도구 분할 연구들은 주로 특정 데이터셋에 최적화된 세그멘테이션 네트워크를 사용했으나, 본 논문은 최신 파운데이션 모델인 SAM의 강력한 분할 능력을 활용하면서도 DETR의 자동 탐지 능력을 결합함으로써 수동 프롬프트의 필요성을 제거했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

Surgical-DeSAM은 크게 **Swin-DETR**와 **Decoupling SAM (DeSAM)** 두 부분으로 구성된다. 전체 흐름은 `입력 이미지 $\rightarrow$ Swin-Transformer $\rightarrow$ DETR Encoder $\rightarrow$ DETR Decoder $\rightarrow$ 바운딩 박스 생성 $\rightarrow$ SAM Prompt Encoder $\rightarrow$ SAM Mask Decoder $\rightarrow$ 최종 분할 마스크` 순으로 진행된다.

### 주요 구성 요소 및 역할

1. **Swin-DETR**:
    - 기존 DETR의 ResNet50을 **Swin-Transformer**로 교체하였다.
    - Swin-Transformer는 Shifted Window 기반의 계층적 Transformer 구조를 사용하여 효율적인 Self-attention 계산이 가능하다.
    - 특히 ResNet50은 특성 맵(Feature map) $f_{resnet} \in \mathbb{R}^{d \times H \times W}$를 시퀀스 형태인 $f \in \mathbb{R}^{d \times HW}$로 변환하는 과정이 필요하지만, Swin-Transformer는 직접적으로 $f_{swin} \in \mathbb{R}^{d \times HW}$ 형태의 출력을 생성하여 DETR 인코더에 바로 입력될 수 있다.

2. **Decoupling SAM (DeSAM)**:
    - SAM의 매우 무거운 이미지 인코더(Image Encoder)를 제거하고, 대신 DETR 인코더의 출력을 직접 SAM의 마스크 디코더(Mask Decoder)에 입력한다.
    - DETR 디코더에서 예측된 바운딩 박스는 SAM의 프롬프트 인코더(Prompt Encoder)를 통해 임베딩 벡터로 변환되어 마스크 디코더에 전달된다.

### 학습 절차 및 손실 함수

본 모델은 탐지와 분할 작업을 동시에 수행하는 End-to-End 방식으로 학습된다. 학습 시에는 바운딩 박스의 Ground-truth와 분할 마스크의 Ground-truth를 모두 사용한다.

전체 손실 함수 $Loss_{total}$은 다음과 같이 정의된다:
$$Loss_{total} = L_{box} + L_{dsc}$$

- $L_{box}$: 탐지 작업을 위한 손실 함수로, GIoU(Generalized Intersection over Union) 손실과 $l_1$ 손실을 결합하여 사용한다.
- $L_{dsc}$: 분할 작업을 위한 손실 함수로, Dice coefficient similarity (DSC) 손실을 사용하여 예측 마스크와 실제 마스크 간의 유사도를 측정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MICCAI 수술 도구 분할 챌린지의 공개 데이터셋인 **EndoVis 2017** 및 **EndoVis 2018**을 사용하였다.
- **지표**: mIoU(mean Intersection over Union)와 DICE score를 사용하여 성능을 측정하였다.
- **최적화**: AdamW 옵티마이저를 사용하였으며, 학습률(Learning rate)은 $10^{-4}$, Weight decay는 $0.1$로 설정하였다.

### 정량적 결과

Surgical-DeSAM은 기존의 SOTA(State-of-the-art) 모델들과 비교하여 성능이 크게 향상되었음을 보였다.

- **EndoVis 2017**: mIoU $82.41$, DICE $89.62$를 달성하였다.
- **EndoVis 2018**: mIoU $84.91$, DICE $90.70$을 달성하였다.
- 이는 TraSeTR, ISINet 등 기존 모델들보다 유의미하게 높은 수치이며, 특히 DICE 지표에서 뚜렷한 우위를 보였다.

### 절제 연구 (Ablation Study)

백본 네트워크에 따른 성능 차이를 분석한 결과, Swin-Transformer가 ResNet50보다 우수한 성능을 보였다.

- 탐지 작업(Detection)에서 mAP@0.50:0.95 기준으로 DETR-SwinB($64.6$)가 DETR-R50($61.4$)보다 높았다.
- 최종 Surgical-DeSAM 모델에서도 Swin-Transformer 백본을 사용했을 때 ResNet50 대비 탐지 mAP는 $2.7\%$, 분할 DICE score는 $7.1\%$ 향상되었다.

## 🧠 Insights & Discussion

본 논문은 SAM의 강력한 분할 능력을 활용하면서도, 의료 현장에서의 실용성을 가로막는 '수동 프롬프트' 문제를 DETR와의 결합을 통해 성공적으로 해결하였다. 특히 SAM의 무거운 이미지 인코더를 제거하고 DETR 인코더를 공유하는 'Decoupling' 전략은 연산 효율성을 높이는 동시에 자동화를 가능하게 한 핵심적인 설계라고 판단된다.

다만, 논문에서 명시적으로 언급되지 않은 부분은 실시간 추론 속도(FPS)에 대한 구체적인 수치이다. "Real-time"이라는 표현을 사용하였으나, 정확히 초당 몇 프레임을 처리할 수 있는지에 대한 정량적 분석이 부족하다. 또한, 수술 도구의 종류가 매우 다양하거나 가려짐(Occlusion)이 심한 환경에서의 강건성(Robustness)에 대해서는 향후 연구 과제로 남겨두고 있다.

비판적으로 해석하자면, 본 연구는 SAM의 마스크 디코더와 프롬프트 인코더를 유지한 채 인코더만 교체한 형태이므로, 여전히 SAM의 디코더 구조에 의존적이다. 따라서 매우 작은 도구의 미세한 분할이나 극심한 노이즈 환경에서 DETR의 바운딩 박스 예측 오류가 분할 결과에 직접적인 영향을 미치는 취약점이 존재할 수 있다.

## 📌 TL;DR

본 논문은 수술 도구 분할을 위해 SAM의 수동 프롬프트 입력 문제를 해결한 **Surgical-DeSAM**을 제안한다. Swin-Transformer 기반의 DETR를 통해 자동으로 바운딩 박스를 생성하고, SAM의 무거운 이미지 인코더를 DETR 인코더로 대체하여 효율적인 End-to-End 분할 파이프라인을 구축하였다. 실험 결과 EndoVis 17/18 데이터셋에서 DICE score 약 90% 내외의 높은 성능을 기록하며 기존 SOTA 모델들을 상회하였다. 이 연구는 수동 개입 없는 자동화된 수술 도구 분할의 가능성을 제시하며, 향후 실시간 수술 보조 시스템의 핵심 모듈로 활용될 가능성이 높다.
