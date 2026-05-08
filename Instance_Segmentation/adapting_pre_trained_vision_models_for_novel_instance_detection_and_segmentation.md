# Adapting Pre-Trained Vision Models for Novel Instance Detection and Segmentation

Yangxiao Lu, Jishnu Jaykumar P, Yunhui Guo, Nicholas Ruozzi, Yu Xiang (2024/2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Novel Instance Detection and Segmentation (NIDS)**이다. NIDS는 각 객체 인스턴스에 대해 단 몇 개의 예시(few examples)만 주어진 상태에서, 쿼리 이미지 내의 새로운 객체 인스턴스를 탐지하고 세그멘테이션하는 작업이다. 이는 특히 로봇 지각(robot perception) 분야에서 매우 중요한데, 로봇이 사전 학습되지 않은 특정 객체를 소수의 템플릿 이미지만으로 인식하고 정확한 Bounding Box와 Mask를 확보하여 파지(grasping)와 같은 작업을 수행해야 하기 때문이다.

기존의 접근 방식들은 다음과 같은 한계를 가지고 있다. 첫째, SAM(Segment Anything Model)과 같은 Open-world Detector를 사용할 경우, 실제 객체가 아닌 단순한 '영역(region)' 단위의 제안(proposal)을 생성하는 경향이 있어, 하나의 객체가 여러 개로 분할되거나 배경이 객체로 오인되는 False Alarm 문제가 발생한다. 둘째, DINOv2의 $\text{cls token}$ 등을 사용한 기존 임베딩 방식은 서로 다른 인스턴스 간의 임베딩이 지나치게 유사하여 변별력이 떨어진다는 문제가 있다.

따라서 본 논문의 목표는 정교한 객체 제안(Object Proposal) 생성 방법과 사전 학습된 비전 모델의 임베딩을 효과적으로 정제하는 메커니즘을 도입하여, 소수의 템플릿만으로도 높은 정확도의 NIDS를 달성하는 **NIDS-Net** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 사전 학습된 거대 비전 모델들의 능력을 결합하고, 이를 최소한의 파라미터 튜닝으로 최적화하는 것이다.

1. **Grounded-SAM을 통한 고품질 제안 생성**: Grounding DINO의 객체성(objectness)과 SAM의 정밀한 마스크 생성 능력을 결합하여, 단순한 영역 제안이 아닌 실제 객체 단위의 Bounding Box와 Mask를 확보함으로써 False Alarm을 획기적으로 줄였다.
2. **Weight Adapter (WA) 도입**: 사전 학습된 임베딩 공간을 파괴하지 않으면서도, 각 인스턴스의 변별력을 높이기 위해 가중치를 학습하는 Weight Adapter를 제안한다. 이는 기존의 CLIP-Adapter와 같은 잔차(residual) 방식이 소수 샘플 환경에서 오버피팅을 유발하여 비대상 객체를 대상으로 오분류하는 문제를 해결한다.
3. **유연한 프레임워크 구성**: DINOv2의 Foreground Feature Averaging (FFA) 파이프라인과 Weight Adapter를 결합하여, 추가적인 대규모 학습 없이도 새로운 인스턴스에 빠르게 적응할 수 있는 통합 프레임워크를 구축하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 차별점을 제시한다.

- **사전 학습 모델 (Pretrained Models)**: DINOv2와 같은 모델들이 강건한 시각적 특징을 제공하지만, 이를 NIDS와 같은 특정 태스크에 효과적으로 활용하는 방법론이 필요함을 강조한다.
- **인스턴스 탐지 (Instance Detection)**: 2D 기반 매칭 방법론들은 폐색(occlusion)이나 포즈 변화에 취약하며, VoxDet와 같은 3D 복셀 표현 방식은 기하학적 불변성을 제공하지만 2D 기반의 강건한 임베딩 생성 필요성이 여전함을 지적한다.
- **어댑터 (Adapters)**: CLIP-Adapter와 같은 기존 방식은 새로운 특징 벡터를 더하는 방식($\text{residual vector}$)을 사용한다. 반면, 본 논문의 Weight Adapter는 기존 특징 공간 내에서 가중치를 조절하는 방식을 취한다. 또한, SENets의 SE-block과 유사해 보일 수 있으나, 본 연구의 어댑터는 Sigmoid 함수 이전에 ReLU 레이어를 배치하여 출력 범위를 $[0.5, 1)$으로 제한함으로써 원본 임베딩과의 근접성을 유지한다는 구조적 차이점이 있다.

## 🛠️ Methodology

NIDS-Net의 전체 파이프라인은 크게 네 가지 단계로 구성된다.

### 1. 인스턴스 임베딩 생성 단계 (Instance Embedding Generation)

각 인스턴스에 대해 $K$개의 템플릿 이미지와 마스크가 주어진다. DINOv2 ViT 백본을 통해 패치 임베딩을 추출한 후, 마스크 영역에 해당하는 전경 특징들의 평균을 구하는 **Foreground Feature Averaging (FFA)**를 적용하여 초기 템플릿 임베딩 $E^T \in \mathbb{R}^{N \times K \times C}$를 생성한다.

### 2. 객체 제안 단계 (Object Proposal Stage)

쿼리 이미지에서 객체 후보를 추출하기 위해 **Grounded-SAM (GS)**을 사용한다.

- **Grounding DINO**: "objects"라는 텍스트 프롬프트를 사용하여 전경 객체의 Bounding Box를 먼저 확보한다.
- **SAM**: 확보된 Bounding Box 내부에서 정밀한 Mask를 생성한다.
이후 생성된 제안 영역들에 대해 FFA 파이프라인을 적용하여 제안 임베딩 $E^P \in \mathbb{R}^{Q \times C}$를 계산한다.

### 3. 어댑터를 통한 임베딩 정제 (Embedding Refinement via Adapter)

본 논문의 핵심인 **Weight Adapter (WA)**는 소수의 템플릿 이미지만을 사용하여 $\text{InfoNCE loss}$로 학습된다.

**Weight Adapter의 동작 원리:**
임베딩 $f$가 주어졌을 때, 다음과 같은 과정을 통해 가중치 $w$를 생성하고 적용한다.
$$w = \text{sigmoid}(\text{MLP}(\beta f))$$
$$f^w = w \odot (\beta f)$$
여기서 $\beta$는 특징 스케일링을 위한 상수(본 논문에서는 10으로 설정)이며, $\odot$은 원소별 곱셈(element-wise multiplication)을 의미한다. ReLU 활성화 함수를 통해 가중치 $w$의 범위를 $[0.5, 1)$으로 제한함으로써, 적응 후의 임베딩이 원본 값에서 크게 벗어나지 않도록 하여 오버피팅을 방지한다.

**유사도 계산:**
정제된 임베딩 $q'$와 $k'$ 사이의 코사인 유사도는 다음과 같이 계산된다.
$$\cos(q', k') = \frac{\sum_{i} w_{1,i} w_{2,i} q_i k_i}{\sqrt{\sum_{i} w_{1,i}^2 q_i^2} \sqrt{\sum_{i} w_{2,i}^2 k_i^2}}$$
이 식을 통해 Weight Adapter는 유사도 계산 시 가장 중요한 임베딩 채널에 더 높은 가중치를 부여하여 인스턴스 간 변별력을 높인다.

### 4. 매칭 단계 (Matching Stage)

제안 임베딩 $E^P$와 템플릿 임베딩 $E^T$ 간의 코사인 유사도를 계산한다. 각 인스턴스의 $K$개 템플릿 점수 중 $\text{Max}$ 값을 취해 최종 인스턴스 점수 행렬($Q \times N$)을 생성한다.

- **세그멘테이션 보완**: 성능 향상을 위해 SAM-6D에서 제안된 외형 매칭 점수 $s_{\text{appe}}$를 추가로 합산하여 최종 점수를 산출한다.
- **라벨 할당**: 객체가 유일하다는 가정하에 **Stable Matching** 알고리즘을 사용하거나, 중복 허용 시 $\text{Argmax}$ 함수를 통해 각 제안에 인스턴스 ID를 할당한다.

## 📊 Results

### 실험 설정

- **데이터셋**: High-resolution (실제 이미지), RoboTools (실제 이미지), LM-O 및 YCB-V (합성 템플릿/실제 테스트), BOP Challenge (7개 핵심 데이터셋).
- **지표**: Average Precision (AP), $\text{AP}_{50}$, $\text{AP}_{75}$.
- **구현**: RTX A5000 GPU, DINOv2 ViT-L 백본 사용.

### 주요 결과

1. **객체 탐지 (Detection)**:
    - **High-resolution 데이터셋**: 기존 최상위 베이스라인보다 $\text{AP}$ 기준 17.7 포인트 높았으며, Weight Adapter를 적용했을 때 $\text{AP}$가 63.9까지 상승하였다.
    - **RoboTools 데이터셋**: SOTA 모델인 VoxDet 대비 46.2 $\text{AP}$라는 압도적인 성능 향상을 보였다.
    - **LM-O 및 YCB-V**: 각각 10.3 $\text{AP}$와 24.0 $\text{AP}$의 향상을 기록하며 VoxDet를 상회하였다.

2. **인스턴스 세그멘테이션 (Segmentation)**:
    - BOP Challenge의 7개 데이터셋에서 RGB 기반 방식 중 최상위 성능을 보였으며, RGB-D 정보를 사용하는 SAM-6D와 비교해서도 경쟁력 있는 성능을 입증하였다 ($\text{Mean AP } 48.6$).

3. **정성적 및 실세계 테스트**:
    - RealSense D455 카메라로 촬영한 실물 객체들에 대해 강건한 탐지 성능을 보였다.
    - Grounded-SAM이 기존 SAM 단독 사용 시 발생하는 배경 오분류 문제를 효과적으로 해결함을 시각적으로 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 거대 사전 학습 모델들을 단순 결합하는 것에 그치지 않고, **Weight Adapter**라는 경량 구조를 통해 '특징 공간의 보존'과 '변별력 강화'라는 두 마리 토끼를 잡았다. 특히 $\text{residual}$ 방식의 어댑터가 가지는 오버피팅 위험을 가중치 곱셈 방식으로 해결하여, 소수의 템플릿만으로도 일반화 성능을 높인 점이 매우 인상적이다. 또한, Grounded-SAM의 도입으로 제안 단계에서의 노이즈를 줄여 전체 파이프라인의 효율성과 정확도를 동시에 높였다.

### 한계 및 비판적 해석

- **계산 자원**: Grounding DINO, SAM, DINOv2 등 여러 개의 거대 모델을 순차적으로 사용하므로, 단일 엔드-투-엔드(end-to-end) 검출기에 비해 추론 시 계산 자원 소모가 클 가능성이 높다.
- **유사 객체 취약성**: 외형이 매우 유사한 인스턴스들이 함께 존재할 경우 오분류가 발생하며, 심한 폐색(occlusion)이 있는 경우 신뢰도 점수가 낮아져 탐지에 실패하는 사례가 보고되었다.
- **템플릿 의존성**: 현재는 $K$개의 템플릿 임베딩을 사용하지만, 실제 환경에서는 단 하나의 이미지로 인식하는 One-shot 능력이 더 중요할 수 있다.

## 📌 TL;DR

본 논문은 사전 학습된 비전 모델들을 활용한 새로운 인스턴스 탐지 및 세그멘테이션 프레임워크인 **NIDS-Net**을 제안한다. **Grounded-SAM**을 통해 고품질의 객체 제안을 생성하고, **Weight Adapter**를 통해 DINOv2 임베딩의 변별력을 높임으로써 오버피팅 없이 소수 샷(few-shot) 환경에서 뛰어난 성능을 달성하였다. 이 연구는 특히 로봇의 실시간 객체 인식 및 파지 작업과 같이, 새로운 환경에서 빠르게 미지의 객체를 식별해야 하는 응용 분야에 매우 중요한 기여를 할 것으로 기대된다.
