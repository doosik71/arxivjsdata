# OBSeg: Accurate and Fast Instance Segmentation Framework Using Segmentation Foundation Models with Oriented Bounding Box Prompts

Zhen Zhou, Junfeng Fan, Yunkai Ma, Sihan Zhao, Fengshui Jing, and Min Tan (2024)

## 🧩 Problem to Solve

원격 탐사 이미지(Remote Sensing Images)에서의 인스턴스 분할(Instance Segmentation)은 객체들이 다양한 크기로 존재하고, 조밀하게 배치되어 있으며, 임의의 방향으로 회전되어 있다는 특성 때문에 매우 까다로운 과제이다. 기존의 수평 바운딩 박스(Horizontal Bounding Box, HBB) 기반 방식은 객체가 기울어져 있을 경우 박스 내부에 불필요한 배경이나 다른 객체가 많이 포함되는 간섭 문제(Interference)가 발생한다.

이를 해결하기 위해 회전 바운딩 박스(Oriented Bounding Box, OBB)를 사용하는 방법들이 제안되었으나, 대부분의 기존 연구들은 "바운딩 박스 내부에서 분할(segmentation within bounding box)"하는 패러다임을 따른다. 이 방식은 마스크 생성 과정이 검출된 OBB 영역 내로 국한되기 때문에, OBB 검출 성능에 지나치게 의존한다는 치명적인 약점이 있다. 즉, OBB 검출이 부정확하면 최종 마스크 결과도 함께 저하되는 불안정성을 보인다. 본 논문의 목표는 OBB를 단순한 제약 영역이 아닌 '프롬프트(Prompt)'로 활용함으로써 검출 성능에 대한 의존도를 낮추고, 정확하고 빠른 인스턴스 분할 프레임워크인 OBSeg를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Segment Anything Model(SAM)과 같은 세그멘테이션 기반 파운데이션 모델(Segmentation Foundation Models, BSMs)의 강력한 마스크 생성 능력과 프롬프트 메커니즘을 OBB에 접목하는 것이다.

주요 기여 사항은 다음과 같다.

1. **OBB 프롬프트 기반 프레임워크**: OBB를 분할의 경계가 아닌 가이드(Prompt)로 사용하여, OBB 검출이 다소 부정확하더라도 정확한 마스크를 생성할 수 있는 OBSeg를 제안하였다.
2. **OBB Prompt Encoder 설계**: 기존 BSM들이 HBB 프롬프트만 지원하는 한계를 극복하기 위해, OBB의 위치와 방향 정보를 효과적으로 인코딩할 수 있는 새로운 OBB Prompt Encoder를 설계하였다.
3. **Gaussian Smoothing 기반 지식 증류(Knowledge Distillation)**: 거대 모델인 BSM의 계산 복잡도를 줄이기 위해 경량 모델로 지식을 전이하며, 이때 Teacher 모델의 출력에 Gaussian Smoothing을 적용하여 Student 모델의 일반화 성능과 분할 정확도를 향상시켰다.

## 📎 Related Works

### 1. OBB 기반 인스턴스 분할

기존의 ISOP, Rotated Blend Mask R-CNN 등은 OBB를 사용하여 위치 정확도를 높였으나, 앞서 언급한 "바운딩 박스 내 분할" 방식에 의존하여 검출기 성능에 종속적인 한계가 있었다.

### 2. 박스 프롬프트 기반 BSMs

SAM과 같은 모델은 대규모 데이터로 사전 학습되어 뛰어난 제로샷 성능을 보이며 HBB 프롬프트를 지원한다. 하지만 원격 탐사 이미지처럼 객체가 조밀하고 회전된 환경에서는 HBB 프롬프트가 많은 간섭 영역을 포함하게 되어 성능이 저하된다.

### 3. 기존 프롬프트 인코더 및 지식 증류의 한계

기존 HBB 인코더는 두 개의 점(좌상단, 우하단)만 사용하므로 회전 정보를 담지 못한다. 또한, EfficientSAM이나 MobileSAM 같은 경량화 모델들은 지식 증류를 사용하지만, Teacher 모델과 Student 모델 간의 성능 격차가 여전히 크다는 문제가 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

OBSeg는 크게 네 가지 모듈로 구성된다: **OBB Detection Module $\rightarrow$ Image Encoder $\rightarrow$ OBB Prompt Encoder $\rightarrow$ Mask Decoder**.
먼저 OBB 검출기가 인스턴스를 구분하고 대략적인 위치와 클래스를 식별한다. 이후 Image Encoder가 이미지 임베딩을 생성하고, OBB Prompt Encoder가 OBB 정보를 임베딩으로 변환한다. 마지막으로 Mask Decoder가 이 두 임베딩을 결합하여 최종 마스크를 예측한다.

### 2. Oriented Bounding Box Prompt Encoder

OBB는 $(x_1, y_1, x_2, y_2, \sin\theta, \cos\theta)$로 파라미터화된다. 여기서 $(x_1, y_1)$과 $(x_2, y_2)$는 각각 좌상단과 우하단 좌표이며, $(\sin\theta, \cos\theta)$는 방향 정보이다.

**Adaptive Gaussian Positional Encoding (AGPE)**:
좌표 기반의 MLP는 고주파 정보를 학습하는 데 어려움이 있다. 이를 해결하기 위해 본 논문은 AGPE를 도입하여 좌표 세트를 랜덤 푸리에 특성(Random Fourier Features)으로 매핑한다.

$$ \gamma(\tau) = [\cos(2\pi G^T \tau), \sin(2\pi G^T \tau)]^T $$

여기서 $G$는 가우시안 분포에서 샘플링된 주파수 행렬이다. 특히 $\Sigma$ (공분산 행렬)를 학습 가능한 파라미터로 설정하여 데이터셋에 적응적으로 대응하게 하였다.

**방향 보정 및 최종 임베딩**:
방향 정보 $(\sin\theta, \cos\theta)$는 $0^\circ$와 $180^\circ$ 부근에서 급격한 값의 변화가 발생하는 경계 불연속성(Boundary Discontinuity) 문제가 있다. 이를 완화하기 위해 학습 가능한 방향 보정 임베딩 $\omega_\theta$를 도입한다. 최종 OBB 프롬프트 임베딩 $P_{OBB}$는 다음과 같이 결합된다.

$$ P_{OBB} = \text{Concat}[E(\phi_p), E(\theta)] $$

### 3. Knowledge Distillation (KD)

Teacher 모델(ViT-H 기반 $\text{OBSeg}^*$)의 지식을 Student 모델(ViT-Tiny 기반 $\text{OBSeg}$)에게 전달하여 경량화를 달성한다.

**Gaussian Smoothing (GS)**:
단순한 모방이 아니라, Teacher의 출력에 가우시안 커널을 컨볼루션하여 부드럽게 만든 타겟을 학습하게 함으로써 일반화 성능을 높인다.

- **프롬프트 증류**: 1차원 가우시안 커널 $G^1_k$를 사용하여 Teacher의 프롬프트 임베딩을 스무딩하고 MSE 손실 함수를 통해 학습한다.
- **마스크 증류**: 2차원 가우시안 커널 $G^2_{k \times k}$를 사용하여 Teacher가 생성한 확률 맵(Probability Map)을 스무딩하고, 이를 타겟으로 BCE 손실 함수를 사용한다.

최종 손실 함수는 다음과 같다.
$$ L_{total} = \lambda \cdot L_{prompt} + (1 - \lambda) \cdot L_{mask} $$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: iSAID (항공 이미지), NWPU VHR-10 (고해상도 광학 이미지), PSeg-SSDD (SAR 선박 이미지).
- **기준 모델**: Oriented R-CNN (OBB 검출), SAM (BSM 기반).
- **지표**: $\text{AP}$, $\text{AP}_{50}$, $\text{AP}_{75}$ 및 추론 속도(FPS).

### 2. 주요 결과

- **정확도**: iSAID 데이터셋에서 $\text{OBSeg}^*$(Teacher)는 $\text{AP} 45.1\%$, $\text{OBSeg}$(Student)는 $\text{AP} 43.7\%$를 기록하며 기존의 HBB 기반 SAM 및 OBB 기반의 다른 방법들을 크게 상회하였다.
- **효율성**: ViT-Tiny 기반의 $\text{OBSeg}$는 $20.4\text{ FPS}$의 경쟁력 있는 추론 속도를 보이면서도 높은 정확도를 유지하였다.
- **강건성**: 시각화 결과, OBB 검출이 다소 부정확하더라도 프롬프트 기반 방식 덕분에 마스크는 정확하게 생성되는 것을 확인하였다. 이는 "바운딩 박스 내 분할" 방식보다 훨씬 유연함을 의미한다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 OBB를 '제약 조건'이 아닌 '힌트(Prompt)'로 재정의함으로써 인스턴스 분할의 패러다임을 전환하였다. 특히 AGPE를 통해 저차원 좌표 정보의 고주파 특성을 잘 살렸으며, 가우시안 스무딩 기반의 지식 증류를 통해 성능 손실을 최소화하며 모델을 경량화한 점이 돋보인다.

### 한계 및 향후 과제

논문에서는 가로세로비(Aspect Ratio)가 매우 크거나 규모(Scale)가 극단적인 객체에 대해서는 성능이 떨어지는 경향이 있다고 명시하였다. 이는 데이터셋 내의 롱테일(Long-tail) 분포 문제로 인한 것으로, 향후 클래스 균형 샘플링 전략이나 극단적 스케일 특징 학습 등을 통해 개선할 필요가 있다.

## 📌 TL;DR

본 논문은 원격 탐사 이미지의 회전된 객체를 정밀하게 분할하기 위해 **OBB를 프롬프트로 사용하는 인스턴스 분할 프레임워크 OBSeg**를 제안하였다. 새로운 **OBB Prompt Encoder**와 **Gaussian Smoothing 기반 지식 증류**를 통해, OBB 검출 성능에 대한 과도한 의존도를 해결함과 동시에 추론 속도를 획기적으로 높였다. 이 연구는 파운데이션 모델을 특정 도메인(원격 탐사)의 특수 프롬프트(OBB)에 맞게 최적화하는 효과적인 방법을 제시하였다는 점에서 향후 연구에 중요한 시사점을 준다.
