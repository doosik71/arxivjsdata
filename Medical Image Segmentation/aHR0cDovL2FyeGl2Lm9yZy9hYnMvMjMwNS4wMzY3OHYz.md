# Towards Segment Anything Model (SAM) for Medical Image Segmentation: A Survey

Yichi Zhang, Rushi Jiao (2023)

## 🧩 Problem to Solve

본 논문은 자연어 처리와 이미지 생성 분야에서 혁신을 일으킨 Foundation Model의 흐름이 이미지 세그멘테이션 영역으로 확장된 Segment Anything Model (SAM)을 의료 영상 세그멘테이션(Medical Image Segmentation, MIS)에 어떻게 적용하고 최적화할 수 있는지를 분석한다.

의료 영상 세그멘테이션은 장기, 병변, 조직 등 특정 해부학적 구조를 구분하는 작업으로, 컴퓨터 보조 진단 및 치료 계획 수립에 필수적이다. 기존의 딥러닝 모델들은 특정 모달리티(modality)나 타겟에 최적화되어 있어 범용적인 일반화 능력이 부족하다는 한계가 있었다. SAM은 강력한 Zero-shot 일반화 능력을 갖추고 있어 이를 해결할 대안으로 주목받았으나, 자연 영상(Natural Image)과 의료 영상 사이의 현저한 차이(구조적 복잡성, 낮은 대비, 관찰자 간 변동성 등)로 인해 SAM을 의료 분야에 직접 적용했을 때의 유효성과 성능에 대한 명확한 검증이 필요했다. 따라서 본 연구의 목표는 SAM의 의료 영상 적용 사례를 체계적으로 정리하고, 성능 향상을 위한 방법론적 적응 방안을 제시하며, 향후 연구 방향을 논의하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 SAM을 의료 영상 세그멘테이션에 적용하려는 최근의 연구 흐름을 두 가지 주요 관점에서 분류하고 분석한 것이다.

첫째, SAM의 Zero-shot 성능을 다양한 의료 영상 작업에서 벤치마킹한 연구들을 종합하여, SAM이 어떤 조건(모달리티, 프롬프트 종류, 객체 특성)에서 효과적이고 어디에서 한계를 보이는지를 명확히 규명하였다.
둘째, SAM의 구조적 한계를 극복하기 위해 제안된 다양한 방법론적 적응(Methodological Adaptations) 방식을 체계화하였다. 여기에는 파라미터 효율적 미세 조정(Parameter-efficient Fine-tuning), 2D에서 3D로의 확장, 프롬프트 생성 자동화, 그리고 SAM의 출력을 입력 데이터로 활용하는 데이터 증강 기법 등이 포함된다.

## 📎 Related Works

논문은 먼저 Foundation Model의 개념을 설명하며, GPT 시리즈와 같이 대규모 데이터로 학습되어 다양한 하위 작업으로 전이 가능한 모델들의 성공 사례를 언급한다. 특히 컴퓨터 비전 분야에서는 CLIP과 같이 시각적 개념과 텍스트 간의 상호작용을 학습한 모델들이 기반이 되었음을 설명한다.

SAM의 경우, SA-1B라는 거대한 데이터셋을 통해 학습되어 프롬프트 기반의 유연한 세그멘테이션이 가능하다는 점이 기존 모델과의 차별점이다. 하지만 의료 영상 분야의 기존 연구들은 특정 장기나 질환에 특화된 전용 모델(Specialist Models) 위주였으며, 이는 높은 정확도를 보장하지만 새로운 데이터셋에 대한 일반화 능력이 떨어진다는 한계가 있었다. 본 논문은 이러한 전용 모델과 SAM과 같은 범용 모델 사이의 간극을 메우기 위한 연구들을 다룬다.

## 🛠️ Methodology

### Segment Anything Model (SAM) 아키텍처
SAM은 크게 세 가지 구성 요소로 이루어져 있다.

1. **Image Encoder**: Masked Auto-encoder(MAE)로 사전 학습된 Vision Transformer (ViT)를 사용하여 $1024 \times 1024$ 해상도의 이미지를 처리하고 $64 \times 64$ 크기의 Image Embedding을 생성한다.
2. **Prompt Encoder**: 사용자의 인터랙션을 임베딩한다.
    - **Sparse Prompts**: 점(point), 박스(box), 텍스트(text)를 처리하며, 점과 박스는 위치 인코딩(positional encoding)을 사용하고 텍스트는 CLIP의 텍스트 인코더를 사용한다.
    - **Dense Prompts**: 마스크(mask) 형태의 프롬프트를 컨볼루션을 통해 임베딩하여 Image Embedding과 요소별 합산(element-wise sum)을 수행한다.
3. **Mask Decoder**: 경량화된 구조로, 두 개의 Transformer layer와 동적 마스크 예측 헤드, IoU score 회귀 헤드로 구성된다. 최종적으로 객체의 전체, 부분, 하위 부분에 해당하는 세 가지 마스크를 생성한다.

학습 시에는 Focal Loss와 Dice Loss의 선형 조합을 손실 함수로 사용하여 감독 학습을 수행한다.

### 의료 영상 적용을 위한 적응 방법론
SAM을 의료 영상에 맞게 최적화하기 위해 다음과 같은 전략들이 제안되었다.

- **Fine-tuning (미세 조정)**: 전체 파라미터를 업데이트하는 대신 Mask Decoder만 튜닝하거나(MedSAM), LoRA(Low-Rank Adaptation)와 같은 파라미터 효율적 기법을 사용하여 이미지 인코더의 일부만을 조정한다.
- **Medical SAM Adapter (MSA)**: ViT 블록에 Adapter 모듈을 추가하여 전이 학습을 수행한다. 특히 3D 의료 영상을 위해 Attention 연산을 공간 브랜치(space branch)와 깊이 브랜치(depth branch)로 나누어 슬라이스 간의 상관관계를 학습한다.
- **Auto-Prompting**: 사용자의 수동 입력 없이 프롬프트를 생성하는 보조 프롬프트 인코더를 학습시켜 Fully Automatic 시스템을 구축한다.
- **Robustness Enhancement (DeSAM)**: 마스크 디코더를 프롬프트 관련 IoU 모듈(PRIM)과 프롬프트 불변 마스크 모듈(PIMM)로 분리하여, 잘못된 프롬프트 입력으로 인한 성능 저하를 최소화한다.
- **Input Augmentation (SAMAug)**: SAM의 결과물을 직접 사용하지 않고, SAM이 생성한 세그멘테이션 맵과 경계 맵을 원본 이미지의 추가 채널로 입력하여 기존 U-Net 등의 모델 성능을 높이는 사전 정보(Prior map)로 활용한다.

## 📊 Results

### Zero-shot 성능 평가 결과
다양한 의료 영상 데이터셋에 대한 실험 결과, SAM의 직접적인 적용은 다음과 같은 특성을 보였다.

- **긍정적 결과**: 뇌 MRI 세그멘테이션에서는 기존의 Gold Standard인 BET(Brain Extraction Tool)와 비슷하거나 더 나은 성능을 보였다. 또한, 경계가 명확하고 큰 객체의 경우 Bounding Box 프롬프트를 사용했을 때 경쟁력 있는 성능을 나타냈다.
- **부정적 결과**: 폴립(Polyp) 세그멘테이션에서는 SOTA 모델 대비 Dice Similarity Coefficient (DSC)가 $14.4\% \sim 36.9\%$ 감소하였다. 복강 CT 장기 세그멘테이션에서도 점(point) 프롬프트를 사용했을 때 DSC가 $20.3\% \sim 40.9\%$ 낮게 측정되었다.
- **주요 원인**: 의료 영상 특유의 낮은 대비(low-contrast), 불규칙한 모양, 모호한 경계선 등이 SAM의 Zero-shot 능력을 저하시키는 주요 원인으로 분석되었다.

### 적응 방법론 적용 결과
- **Fine-tuning의 효과**: MedSAM과 같은 미세 조정 모델은 2D/3D 세그멘테이션 작업에서 SAM의 기본 성능을 크게 향상시켰으며, 일부 작업에서는 전용 모델에 근접하는 성능을 보였다.
- **데이터 증강의 효과**: SAMAug 기법을 통해 세포 세그멘테이션의 AJI(Aggregated Jaccard Index)를 $58.36\%$에서 $64.30\%$로 향상시켜, SAM의 불완전한 마스크라도 유용한 특징 정보가 될 수 있음을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 SAM이 의료 영상 분야에서 가진 가능성과 명확한 한계를 동시에 제시한다.

**강점 및 가능성**: SAM은 매우 강력한 이미지 임베딩 능력을 갖추고 있으며, 적절한 프롬프트(특히 Box prompt)가 제공될 때 높은 범용성을 보인다. 또한, 전문가가 처음부터 마스크를 그리는 대신 SAM의 초안을 수정하는 방식의 인터랙티브 세그멘테이션을 통해 의료 영상 라벨링 비용을 획기적으로 줄일 수 있는 잠재력이 크다.

**한계 및 비판적 해석**:
1. **데이터 도메인의 차이**: SAM이 학습한 SA-1B 데이터셋의 자연 영상은 엣지 정보가 뚜렷하지만, 의료 영상은 그렇지 않다. 이는 단순한 튜닝을 넘어 의료 영상 특화 Foundation Model의 필요성을 시사한다.
2. **차원 확장 문제**: 대부분의 의료 영상은 3D 볼륨 데이터이지만 SAM은 2D 기반이다. 단순히 슬라이스별로 처리하는 방식은 슬라이스 간 연속성을 무시하므로, 3D 컨텍스트를 통합하는 구조적 개선이 필수적이다.
3. **프롬프트 의존성**: 성능이 프롬프트의 질에 너무 민감하게 반응하며, 특히 의료 분야에서 요구하는 극도로 높은 정밀도를 달성하기에는 Zero-shot 성능만으로는 부족하다.

## 📌 TL;DR

본 논문은 이미지 세그멘테이션의 기반 모델인 SAM을 의료 영상 분야에 적용하려는 최신 연구들을 종합적으로 분석한 서베이 논문이다. SAM은 일부 작업에서 뛰어난 성능을 보이지만, 의료 영상의 특성상 직접 적용 시 한계가 명확하며, 이를 해결하기 위해 **Parameter-efficient Fine-tuning, 3D Adapter 도입, 프롬프트 자동화** 등의 적응 전략이 필요함을 강조한다. 이 연구는 향후 의료 영상 전용의 대규모 Foundation Model 구축과 임상 워크플로우 통합을 위한 중요한 가이드라인을 제공한다.