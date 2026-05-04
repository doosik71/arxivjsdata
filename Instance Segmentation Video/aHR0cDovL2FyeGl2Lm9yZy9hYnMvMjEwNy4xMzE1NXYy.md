# Improving Video Instance Segmentation via Temporal Pyramid Routing

Xiangtai Li, Hao He, Yibo Yang, Henghui Ding, Kuiyuan Yang, Guangliang Cheng, Yunhai Tong, Dacheng Tao (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 비디오 인스턴스 분할(Video Instance Segmentation, VIS)에서 발생하는 객체의 크기 변화(scale variation)와 객체 간 겹침(overlapping) 문제이다. VIS는 비디오 시퀀스 내의 각 인스턴스를 동시에 검출(detect), 분할(segment), 그리고 추적(track)해야 하는 복합적인 태스크이다.

기존의 접근 방식들은 주로 단일 프레임 특징이나 여러 프레임의 단일 스케일 특징에 의존하며, 이로 인해 시간적 정보(temporal information)나 다중 스케일 정보(multi-scale information) 중 하나를 간과하는 경향이 있다. 특히 비디오 내에서 객체의 크기가 급격하게 변할 때, 기존 방법론들은 이를 효과적으로 처리하지 못해 모호한 예측 결과를 생성하는 한계가 있다. 따라서 본 연구의 목표는 시간적 정렬과 다중 스케일 특징의 효율적인 집계(aggregation)를 통해 VIS의 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시간적 피라미드 라우팅(Temporal Pyramid Routing, TPR) 전략을 제안하는 것이다. TPR은 두 인접 프레임의 피라미드 특징 쌍으로부터 픽셀 수준의 정렬과 집계를 조건부(conditionally)로 수행한다.

주요 설계 아이디어는 다음과 같다.

1. **Dynamic Aligned Cell Routing (DACR)**: 시간적 차원에서 피라미드 특징을 정렬하고, 게이팅(gating) 메커니즘을 통해 배경 소음을 제거하고 유의미한 시맨틱 정보만을 선택적으로 흡수한다.
2. **Cross Pyramid Routing (CPR)**: 시간적으로 집계된 특징을 스케일 차원에서 동적으로 전달하여 다중 스케일 표현 능력을 강화한다.
3. **Plug-and-Play 모듈**: TPR은 경량화된 구조로 설계되어 기존의 다양한 인스턴스 분할 및 파놉틱 분할(panoptic segmentation) 방법론에 쉽게 결합하여 성능을 높일 수 있다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Instance Segmentation**: Mask R-CNN과 같은 Top-down 방식과 Bottom-up 방식이 존재하며, 본 연구는 주로 Top-down 방식의 프레임워크를 기반으로 한다.
2. **Video Instance Segmentation (VIS)**: MaskProp은 마스크 특징을 전파하여 성능을 높이려 했으나 연산 비용이 높고, VisTR은 트랜스포머를 통해 시퀀스 전체를 처리하지만 다중 스케일의 시간적 정렬 문제는 충분히 다루지 않았다.
3. **Dynamic Network Design**: 이미지 영역에서는 입력 데이터에 따라 네트워크 경로를 변경하는 동적 라우팅(dynamic routing) 연구가 진행되었으나, VIS나 VPS와 같은 비디오 밀집 예측(dense prediction) 작업에 적용된 사례는 부족했다.

### 기존 방식과의 차별점

TPR은 기존의 Optical Flow 기반 워핑(warping) 방식과 달리 별도의 플로우 네트워크 학습이 필요 없으며, 연산 비용이 고정된 어텐션 기반 방식보다 효율적이다. 특히 픽셀 수준의 게이트를 통해 입력 내용에 따라 필요한 특징만 전파하므로, 불필요한 연산을 줄이면서도 시간적 일관성을 유지할 수 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

TPR은 특징 추출기(Backbone + FPN)와 작업 헤드(Task Heads) 사이에 삽입되는 플러그인 모듈이다. 쿼리 프레임(현재 프레임, $X^q_s$)과 참조 프레임(이전 프레임, $X^r_s$)의 특징 피라미드를 입력으로 받아 정제된 특징 피라미드를 출력한다.

### 주요 구성 요소 및 동작 원리

#### 1. Dynamic Aligned Cell Routing (DACR)

DACR은 참조 프레임에서 가장 관련성이 높은 정보를 픽셀 단위로 샘플링하고 필터링한다.

- **샘플링**: Deformable Convolution (DCN)을 사용하여 두 프레임 간의 대응 관계를 찾고 특징을 샘플링한다.
  $$\hat{X}^q_s(i) = \sum_{p_n \in R} X^q_s(i+p_n+O^q_s(p_n)) \cdot W(p_n)$$
  여기서 $O^q_s$는 예측된 오프셋 필드이며, $W$는 커널 가중치이다.
- **이중 게이트(Double Gates) 설계**:
  - **Inner Gate**: 참조 특징에서 세밀한 지원 정보를 찾고 배경 소음을 제거한다.
      $$\hat{Y}^s = \{Y^i_s \mid Y^i_s = H^i_s \cdot \text{Gate}(\text{Cat}(\hat{X}^q_s, \hat{X}^r_s)), i \in q, r\}$$
  - **Outer Gate**: 쿼리 프레임의 정보와 병합된 특징 간의 가중치를 조절하여 최종 특징을 생성한다.
      $$\hat{Y}^{final}_s = G^s \cdot X^q_s + (1-G^s) \cdot Y^{merge}_s$$
      여기서 $G^s$는 시그모이드 함수를 통해 생성된 게이트 맵이다.

#### 2. Cross Pyramid Routing (CPR)

DACR을 통해 정제된 특징을 다른 스케일로 전달하는 과정이다.

- **Bottom-up Routing**: 저해상도 특징이 고해상도 특징으로 전달되는 경로를 설계하여 시맨틱 갭을 메운다. 저해상도에서의 정렬 오류가 더 심하기 때문에, 고해상도 표현을 먼저 전파하는 Bottom-up 방식이 더 안정적이고 빠르다고 설명한다.
- **라우팅 공간**: 기본적으로 $[3, 2, 1, 1]$의 라우팅 깊이 설정을 사용한다.

### 학습 절차 및 손실 함수

학습 시에는 쿼리 프레임 주변에서 참조 프레임을 무작위로 샘플링한다. 전체 손실 함수 $L$은 다음과 같이 정의된다.
$$L = \lambda_1 L_{tasks} + \lambda_2 L_{budget}$$

- $L_{tasks}$: 검출, 분할, 추적 헤드에서 발생하는 작업 관련 손실이다.
- $L_{budget}$: 동적 네트워크의 연산량 증가를 억제하기 위한 예산 손실이다. 게이트 값에 의해 활성화된 픽셀의 수와 전체 연산 복잡도의 비율로 계산된다.

## 📊 Results

### 실험 설정

- **데이터셋**: YouTube-VIS (2019, 2021) 및 Cityscapes-VPS를 사용하였다.
- **지표**: VIS에서는 mAP(mean Average Precision)를, VPS에서는 VPQ(Video Panoptic Quality)를 사용하였다.
- **비교 대상**: MaskProp, VisTR, BlendMask, SipMask 등 최신 모델들을 베이스라인으로 설정하였다.

### 주요 결과

1. **정량적 성과**:
    - BlendMask(ResNet50) 기반으로 TPR을 적용했을 때 mAP가 $33.8\% \rightarrow 36.2\%$로 약 $2.4\%$ 상승하였다.
    - ResNet101 백본 사용 시 $36.6\% \rightarrow 39.1\%$로 $2.5\%$ 상승하여 모델 확장성을 입증하였다.
    - Swin-base 백본 적용 시 $45.9$ mAP를 달성하여, MaskProp보다 더 빠르면서도 더 높은 성능을 보였다.
2. **효율성**:
    - GFlops 증가량이 매우 적으며(평균 $+4.4 \sim 4.9$ GFlops), 추론 속도(FPS)의 하락이 최소화되었다.
3. **VPS 일반화**: Cityscapes-VPS 데이터셋에서도 VPQ 지표가 향상되어, VIS 외의 비디오 장면 이해 작업에도 효과적임을 확인하였다.
4. **정성적 결과**: 객체가 서로 겹치거나(overlapping), 형태가 크게 변하는(deformation) 상황에서도 베이스라인보다 훨씬 견고하게 인스턴스를 추적하고 분할하는 모습을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 시간적 정렬(DACR)과 스케일 간 라우팅(CPR)을 결합함으로써, 비디오 데이터의 고유한 특성인 '시간적 변화'와 '크기 변화'를 동시에 해결하였다. 특히, 모든 픽셀을 처리하는 대신 게이트를 통해 필요한 부분만 선택적으로 전파하는 동적 라우팅 방식이 연산 효율성과 성능이라는 두 마리 토끼를 잡은 핵심 요인으로 분석된다.

### 한계 및 논의사항

- **예산 손실 의존성**: $\lambda_2$ (budget loss) 설정에 따라 효율성과 효과성 사이의 트레이드-오프가 발생한다.
- **참조 프레임 샘플링**: 추론 시에는 바로 이전 프레임(online setting)만 사용하지만, 학습 시에는 무작위 샘플링을 사용한다. 이 간극이 실제 환경에서 어떤 영향을 미치는지에 대한 추가 논의가 필요할 수 있다.
- **Bottom-up vs Top-down**: CPR 설계에서 Bottom-up 방식이 우수함을 보였는데, 이는 저해상도 특징의 정렬 불안정성을 고해상도 특징이 보완해 주기 때문이라는 해석을 제시한다.

## 📌 TL;DR

본 논문은 비디오 인스턴스 분할(VIS)에서 객체의 크기 변화와 겹침 문제를 해결하기 위해 **Temporal Pyramid Routing (TPR)** 모듈을 제안한다. TPR은 **DACR**을 통해 시간적 특징을 정렬하고 소음을 제거하며, **CPR**을 통해 다중 스케일 간 특징을 효율적으로 전파한다. 이 모듈은 기존 모델에 쉽게 추가할 수 있는 **Plug-and-play** 형태이며, 연산량 증가를 최소화하면서도 mAP를 $2 \sim 3\%$ 가량 향상시킨다. 특히 온라인 추론 설정에서 매우 효율적이며, 향후 비디오 장면 이해를 위한 동적 네트워크 설계에 중요한 방향성을 제시한다.
