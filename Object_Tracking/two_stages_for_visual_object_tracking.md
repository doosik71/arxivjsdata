# Two Stages for Visual Object Tracking

Fei Chen, Fuhan Zhang, and Xiaodong Wang (연도 미표기)

## 🧩 Problem to Solve

본 논문은 비디오 시퀀스에서 대상 객체를 정확하게 추적하는 Visual Object Tracking(VOT) 문제를 다룬다. 특히, 기존의 딥러닝 기반 추적기들이 직면한 정밀한 대상 위치 파악(target localization), 크기(scale), 그리고 종횡비(aspect ratio) 추정의 어려움을 해결하고자 한다.

기존의 Siamese 기반 추적기들은 주로 분류(classification)와 바운딩 박스 회귀(bounding box regression)라는 두 개의 분리된 브랜치를 사용한다. 한편, 이미지 세그멘테이션(image segmentation)은 더 정확한 대상 영역을 얻을 수 있는 대안적인 방법으로 제시되었으나, SiamMask와 같은 기존 연구에서는 세그멘테이션 브랜치가 다른 브랜치들과 결합되어 있어 전체 네트워크를 학습시키는 데 어려움이 있다는 한계가 있었다. 따라서 본 논문의 목표는 검출(detection)과 세그멘테이션(segmentation)의 2단계(two-stage) 구조를 통해, 학습의 난이도를 낮추면서도 더 정밀한 추적 결과를 얻는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Siamese 네트워크를 이용한 '거친 위치 추정(coarse state estimation)'과 세그멘테이션 모듈을 이용한 '정밀한 영역 정제(fine refinement)'를 순차적으로 수행하는 2단계 파이프라인을 구축하는 것이다.

1. **Detection-to-Segmentation 구조**: 먼저 Siamese 네트워크를 통해 대상의 대략적인 위치와 크기를 파악하고, 이후 이 정보를 바탕으로 세그멘테이션 마스크를 생성하여 최종적으로 정밀한 회전 바운딩 박스(rotated bounding box)를 도출한다.
2. **Anchor-free Detector 도입**: 계산 복잡도를 줄이고 다양한 형태 변화에 적응하기 위해 Anchor-free 방식의 검출기를 사용하여 타겟의 중심점과 경계까지의 거리를 예측한다.
3. **특징 맵 정제**: 세그멘테이션 단계에서 ResNet-50의 서로 다른 계층에서 추출된 특징 맵들을 결합하여 마스크의 정확도를 높였다.

## 📎 Related Works

### 1. Correlation Filter (CF) 기반 추적기

CF 기반 방법론은 연산 속도가 빠르며, 공간적 정규화(spatial regularization) 등을 통해 성능을 향상시켜 왔다. 최근에는 딥러닝 특징(deep features)을 결합하여 성능을 높이려는 시도가 있었으나, 여전히 딥러닝 기반의 end-to-end 프레임워크에 비해 유연성이 떨어진다는 한계가 있다.

### 2. Siamese 기반 추적기

SiamRPN, SiamRPN++ 등은 템플릿과 검색 영역 간의 상호 상관(cross-correlation)을 이용해 빠르게 대상을 추적한다. 특히 SiamRPN++는 깊은 네트워크(ResNet-50)와 depth-wise cross-correlation을 도입해 성능을 끌어올렸으나, 여전히 정밀한 바운딩 박스 생성에는 한계가 존재한다.

### 3. Segmentation 기반 추적기

SiamMask와 같이 바운딩 박스와 세그멘테이션 마스크를 동시에 예측하는 방식이 제안되었다. 하지만 앞서 언급했듯이 여러 브랜치를 동시에 학습시키는 과정에서 최적화의 어려움이 발생한다. 또한 D3S와 같은 방식은 세그멘테이션 성능이 초기 위치 추정 모듈(GEM)의 성능에 크게 의존한다는 문제점이 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인

전체 시스템은 **Backbone Network $\rightarrow$ Detection Module $\rightarrow$ Segmentation Module** 순으로 구성된다. ResNet-50을 백본으로 사용하여 특징을 추출하며, 1단계에서 검출된 결과를 2단계의 세그멘테이션 입력으로 사용하여 최종 결과를 도출한다.

### 2. Stage 1: Detection Module (Anchor-free)

본 모델은 Anchor-free detector를 사용하여 타겟의 중심 위치와 경계 거리를 예측한다.
특징 맵 $\mathbf{f} \in \mathbb{R}^{H \times W \times i}$의 각 위치 $(x, y)$에 대해, 해당 위치가 ground-truth 바운딩 박스 $\mathcal{B} = \{x_0, y_0, x_1, y_1\}$ 내에 포함되면 양성 샘플(positive sample)로 간주한다.

**학습 목표(Training Objective)**:
타겟 중심 $(x, y)$로부터 바운딩 박스의 상, 하, 좌, 우 경계까지의 거리 $T = \{t^*, l^*, b^*, r^*\}$를 다음과 같이 정의한다.
$$l^* = x - x_0, \quad r^* = x_1 - x, \quad t^* = y - y_0, \quad b^* = y_1 - y$$

**추론 절차(Inference)**:
예측된 값 $\{p_l, p_t, p_r, p_b\}$를 이용하여 최종 바운딩 박스 $\mathcal{B}_{pred} = \{x_{p0}, y_{p0}, x_{p1}, y_{p1}\}$를 다음과 같이 계산한다.
$$x_{p0} = x_c - p_l, \quad x_{p1} = x_c + p_r, \quad y_{p0} = y_c - p_t, \quad y_{p1} = y_c + p_b$$
여기서 $(x_c, y_c)$는 신뢰도 점수가 가장 높은 위치이다.

### 3. Stage 2: Segmentation Module

1단계에서 얻은 바운딩 박스와 특징 맵을 입력으로 받아 이진 세그멘테이션 마스크를 생성한다.

- **구조**: D3S의 프레임워크를 기반으로 하며, `ConvTransposed2D` 모듈을 사용하여 고차원 특징 맵을 저차원 해상도로 업샘플링한다.
- **특징 결합**: 업샘플링 과정에서 ResNet-50의 `conv1`, `layer1`, `layer2`에서 추출된 특징 맵들을 각각 추가하여 정보량을 풍부하게 한다.
- **최종 출력**: Softmax 함수를 통해 생성된 확률 맵에 임계값(threshold) 0.5를 적용하여 이진 마스크를 생성하고, 이 마스크를 기반으로 회전 바운딩 박스(rotated bounding box)를 산출한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: VOT-2016, VOT-2018, VOT-2019, GOT-10k.
- **평가 지표**:
  - VOT: EAO (Expected Average Overlap), Accuracy(A), Robustness(Ro).
  - GOT-10k: AO (Average Overlap), Success Rate (SR).
- **구현**: ResNet-50 백본 사용, 세그멘테이션 네트워크는 Youtube-VOS 데이터셋으로 사전 학습됨.

### 2. 정량적 결과

- **VOT-2016**: EAO 52.6%를 기록하며 비교 대상 중 가장 높은 성능을 보였다. 특히 D3S 대비 EAO가 3.3% 상승하였다.
- **VOT-2018**: EAO 51.3%, Accuracy 66.2%를 달성하여 D3S 및 OceanOn보다 2.4% 높은 EAO를 기록했다.
- **VOT-2019**: EAO 39.0%를 기록하며 D3S(34.8%)와 OceanOn(35.0%)을 크게 상회하였다. 실패율(Failure rate) 또한 27.6%로 감소하여 강건함이 입증되었다.
- **GOT-10k**: AO 0.604를 기록하며 OceanOn(0.611)과 대등한 수준의 경쟁력 있는 성능을 보였으며, D3S보다는 우수한 결과를 나타냈다.

## 🧠 Insights & Discussion

본 연구는 검출과 세그멘테이션을 분리하는 2단계 전략이 실제 추적 성능 향상에 매우 효과적임을 보여주었다. 특히, 기존의 통합형 모델(SiamMask 등)에서 발생하던 학습의 불안정성 문제를 해결함과 동시에, 세그멘테이션의 정밀함을 이용해 VOT-2019와 같은 회전 바운딩 박스 평가 환경에서 탁월한 성과를 거두었다.

다만, 본 논문에서는 1단계 검출기의 성능이 2단계 세그멘테이션의 입력으로 직접 사용되므로, 1단계에서 완전히 실패할 경우 2단계에서도 복구할 방법이 없다는 의존성 문제가 존재할 수 있다. 또한, 2단계 모듈의 추가로 인해 단일 단계 추적기보다 연산 시간이 증가했을 가능성이 있으나, 이에 대한 구체적인 FPS(Frames Per Second) 분석은 본문에 명시되지 않았다.

## 📌 TL;DR

본 논문은 **'검출 $\rightarrow$ 세그멘테이션'**으로 이어지는 2단계 추적 프레임워크를 제안하여, Siamese 네트워크의 빠른 검출 능력과 세그멘테이션의 정밀한 영역 추출 능력을 결합하였다. ResNet-50 백본과 Anchor-free detector를 사용해 학습 효율을 높였으며, VOT-2016, 2018, 2019 벤치마크에서 SOTA 수준의 성능을 달성하였다. 이 연구는 정밀한 객체 경계 추정이 필요한 고정밀 비주얼 트래킹 시스템 설계에 중요한 방향성을 제시한다.
