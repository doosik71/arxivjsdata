# ASIST: Annotation-Free Synthetic Instance Segmentation and Tracking by Adversarial Simulations

Quan Liu et al. (2021)

## 🧩 Problem to Solve

현미경 비디오의 정량적 분석을 위해서는 세포 및 세포 하위 구조(subcellular objects)의 인스턴스 분할(Instance Segmentation)과 추적(Tracking)이 필수적이다. 기존의 전통적인 방식은 각 프레임에서 객체를 분할한 뒤 프레임 간에 객체를 연결하는 2단계(two-stage) 전략을 사용한다. 최근에는 픽셀 임베딩(pixel-embedding) 기반의 딥러닝을 통해 분할과 추적을 동시에 수행하는 단일 단계(single-stage) 솔루션이 등장하였다.

그러나 픽셀 임베딩 기반 학습은 공간적(분할) 및 시간적(추적) 일관성을 모두 갖춘 정밀한 어노테이션(annotation)을 필요로 한다. 현미경 영상은 객체들이 서로 겹치거나 접촉해 있는 밀집 상태(dense objects)이거나, 불규칙한 움직임 및 세포 분열(mitosis)과 같은 높은 역동성(high dynamics)을 보이기 때문에, 사람이 직접 일관된 어노테이션을 생성하는 것은 매우 많은 리소스가 소모되며 확장이 어렵다는 문제가 있다. 따라서 본 논문의 목표는 이러한 수동 어노테이션 없이도 인스턴스 분할과 추적을 수행할 수 있는 annotation-free 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **적대적 시뮬레이션(Adversarial Simulations)**을 통해 실제 데이터와 유사한 합성 데이터와 어노테이션을 생성하고, 이를 통해 단일 단계 픽셀 임베딩 모델을 학습시키는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **ASIST 프레임워크 제안**: 적대적 시뮬레이션(CycleGAN 기반)과 단일 단계 픽셀 임베딩 딥러닝을 결합하여 어노테이션 없이 학습 가능한 파이프라인을 구축하였다.
2.  **어노테이션 정제(Annotation Refinement) 기법**: HeLa 세포와 같이 형태 변화가 심한 객체를 위해, 원(circle)을 중간 표현체로 사용하여 형태의 변형을 시뮬레이션하는 정제 방법을 제안하였다.
3.  **최초의 시도**: 현미경 비디오의 단일 단계 픽셀 임베딩 기반 인스턴스 분할 및 추적 분야에서 어노테이션 프리(annotation-free) 솔루션을 탐색한 첫 번째 연구이다.

## 📎 Related Works

### 이미지 합성 (Image Synthesis)
단순한 이미지 변환(회전, 크기 조정 등)부터 GAN(Generative Adversarial Networks)을 이용한 고해상도 이미지 생성까지 다양한 연구가 진행되었다. 특히, 짝지어지지 않은(unpaired) 데이터셋 간의 변환을 가능하게 하는 CycleGAN은 의료 영상 합성 분야에서 가능성을 보여주었다.

### 현미경 이미지 분할 및 추적
초기에는 강도 기반 임계값 설정(thresholding)이나 Watershed 알고리즘과 같은 전통적인 방식이 사용되었다. 이후 광학 흐름(optical flow)이나 능동 윤곽선(active contours)을 이용한 추적 방법이 제안되었으며, 최근에는 CNN 기반의 지도 학습(supervised learning) 방식이 우수한 성능을 보이고 있다.

### 기존 방식과의 차별점
기존의 픽셀 임베딩 기반 딥러닝 방식은 성능은 좋으나 막대한 양의 수동 어노테이션이 필수적이었다. ASIST는 CycleGAN을 통해 실제 데이터 없이도 학습 가능한 합성 데이터를 생성함으로써 이 의존성을 완전히 제거하였다.

## 🛠️ Methodology

ASIST 프레임워크는 크게 세 단계의 파이프라인으로 구성된다.

### 1. 비지도 이미지-어노테이션 합성 (Unsupervised Image-Annotation Synthesis)
CycleGAN을 사용하여 실제 현미경 이미지와 합성 어노테이션 사이의 상호 변환을 학습한다.
- **형태 모델링**: Microvilli는 막대 모양(stick-shaped), HeLa 세포 핵은 공 모양(ball-shaped)으로 단순화하여 가짜 어노테이션을 생성한다.
- **구조**: Generator A와 B는 모두 9개의 residual block을 가진 ResNet을 인코더로 사용한다.
- **역할**: Generator B는 합성된 어노테이션을 실제 현미경 이미지 스타일로 변환하는 역할을 수행한다.

### 2. 비디오 합성 (Video Synthesis)
학습된 Generator B를 이용하여 '어노테이션 프레임 $\rightarrow$ 비디오' 형태로 확장한다.
- **Microvilli 시뮬레이션**: 객체 수, 이동(translation), 회전, 길이 변화(shortening/lengthening), 진출입(moving in/out) 등을 설정하여 어노테이션 비디오를 만든 후 이를 이미지로 변환한다.
- **HeLa 세포 시뮬레이션**: 더 높은 자유도의 변화를 모델링한다. 여기에는 반지름 변화, 객체의 등장 및 소멸, 특히 세포 분열(mitosis, 모세포가 사라지고 작은 두 개의 딸세포가 생성됨), 객체 간 겹침(overlapping) 등이 포함된다.

### 3. HeLa 세포를 위한 어노테이션 정제 (Annotation Refinement)
단순한 원형(circle) 표현으로는 실제 HeLa 세포의 복잡한 형태를 반영하기 어렵기 때문에 다음 과정을 거친다.
- **이진 마스크 생성**: CycleGAN의 초기 에포크(epoch) 모델(Generator $A^*$)을 사용하여 형태 적응보다는 강도 적응에 집중한 날카로운 이진 마스크(binary mask)를 생성한다.
- **어노테이션 변형(Annotation Deformation, AD)**: ANTs의 비강체 등록(non-rigid registration) 기법을 사용하여 원형 어노테이션을 생성된 이진 마스크의 형태에 맞게 변형시킨다.
- **어노테이션 세척(Annotation Cleaning, AC)**: 변형된 어노테이션과 이진 마스크를 비교하여, 90% 이상 겹치지 않는 불일치 객체를 제거하거나 배경으로 재할당하여 일관성을 높인다.

### 4. 인스턴스 분할 및 추적 (Instance Segmentation and Tracking)
최종적으로 합성된 비디오와 어노테이션을 사용하여 모델을 학습시킨다.
- **백본 네트워크**: RSHN(Recurrent Stacked Hourglass Network)을 사용하여 각 픽셀의 임베딩 벡터를 인코딩한다. RSHN은 시간적 정보를 처리하기 위해 Convolutional GRU를 포함한다.
- **학습 목표**: 동일 객체에 속한 픽셀들은 서로 유사한 임베딩을 가지게 하고, 서로 다른 객체의 픽셀들은 서로 다른 임베딩을 가지도록 유도한다.
- **추론 절차**: 테스트 비디오에서 Faster Mean-shift 알고리즘을 사용하여 픽셀들을 클러스터링함으로써 최종적으로 객체를 분할하고 추적한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Microvilli(자체 데이터) 및 HeLa 세포(ISBI Cell Tracking Challenge의 N2DL-HeLa 데이터셋).
- **평가 지표**: ISBI 챌린드의 표준 지표인 $\text{DET}$(검출), $\text{SEG}$(분할), $\text{TRA}$(추적)를 사용하며, 이는 Acyclic Oriented Graph Matching (AOGM) 알고리즘을 기반으로 측정된다. 값은 $0$에서 $1$ 사이이며 $1$에 가까울수록 성능이 좋다.

### 주요 결과
- **Microvilli 비디오**: 
    - 수동 어노테이션으로 학습한 $\text{RSHN (Real)}$ 모델보다 합성 데이터로 학습한 $\text{ASIST (Microvilli-20)}$ 모델이 더 높은 성능을 보였다.
    - 특히 시뮬레이션된 학습 비디오의 수가 많을수록 성능이 향상되는 경향을 보였다.
- **HeLa 세포 비디오**: 
    - $\text{ASIST (HeLa-AD+AC+HW)}$ 설정이 가장 우수한 성능을 기록하였다.
    - 완전 지도 학습(manual annotation) 기반 모델보다는 약 $5\% \sim 9\%$ 낮은 성능을 보였으나, 전반적으로 대등한 수준의 성능을 달성하였다. 특히 어노테이션 세척(AC) 단계가 성능 향상에 핵심적인 역할을 하였다.

## 🧠 Insights & Discussion

### 강점
본 연구는 딥러닝 모델 학습에 필수적인 대규모의 수동 어노테이션 과정을 완전히 제거하였다. 또한 전통적인 모델 기반 방법들과 달리 하이퍼파라미터 튜닝에 대한 의존도가 낮아 견고함(robustness)을 갖추고 있다. 적대적 학습과 픽셀 임베딩이라는 두 가지 최신 기법을 조화롭게 결합하여 실용적인 솔루션을 제시하였다.

### 한계 및 비판적 해석
현재 제안된 방법은 Microvilli나 HeLa 세포처럼 형태와 외관의 변화가 비교적 균일(homogeneous)한 객체에 최적화되어 있다. 따라서 매우 복잡하고 이질적인 형태를 가진 세포 라인이나 다양한 현미경 영상에 적용하기에는 한계가 있을 수 있다. 논문에서도 언급되었듯이, 더 복잡한 형태를 캡처하기 위해서는 단순한 등록(registration) 기반 방법보다는 Shape Auto-encoder와 같은 딥러닝 기반의 형태 모델링 도입이 필요할 것으로 보인다.

## 📌 TL;DR

ASIST는 수동 어노테이션 없이 현미경 영상의 세포 추적 및 분할을 수행하는 프레임워크이다. CycleGAN을 이용해 가짜 이미지와 어노테이션을 생성하고, 이를 통해 픽셀 임베딩 모델(RSHN)을 학습시킨다. 특히 HeLa 세포의 복잡한 형태를 위해 어노테이션 정제(AD, AC) 과정을 도입하였다. 실험 결과, Microvilli에서는 지도 학습 모델을 능가했고, HeLa 세포에서는 이에 근접하는 성능을 보여, 고비용의 수동 레이블링 문제에 대한 강력한 대안이 될 가능성을 입증하였다.