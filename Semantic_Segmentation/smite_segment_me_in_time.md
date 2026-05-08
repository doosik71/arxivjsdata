# SMITE: SEGMENTMEINTIME

Amirhossein Alimohammadi, Sauradip Nag, Saeid Asgari Taghanaki, Andrea Tagliasacchi, Ghassan Hamarneh, Ali Mahdavi Amiri (2025)

## 🧩 Problem to Solve

본 논문은 비디오 내 객체의 세그멘테이션(segmentation) 문제를 다루며, 특히 **Flexible Granularity**(유연한 입도)라는 개념에 집중한다. Flexible Granularity란 사용자의 목적에 따라 세그멘테이션의 상세 수준(예: 얼굴 전체를 하나로 잡을 것인지, 이마와 눈 등 세부 파트로 나눌 것인지)이 임의로 변할 수 있음을 의미한다.

기존의 비디오 세그멘테이션 방식은 특정 클래스나 고정된 파트 수준에 의존하는 경우가 많으며, 새로운 비디오마다 방대한 양의 어노테이션(annotation)을 요구하는 경향이 있다. 이는 VFX 제작과 같은 실제 산업 현장에서 매우 비효율적이다. 따라서 본 연구의 목표는 **비디오 프레임에 포함되지 않은 소수의 참조 이미지(few reference images)만으로, 새로운 비디오 내 객체를 사용자가 원하는 임의의 입도로 일관되게 세그멘테이션하는 모델을 개발**하는 것이다.

## ✨ Key Contributions

SMITE의 핵심 아이디어는 사전 학습된 **Text-to-Image (T2I) Diffusion Model**의 강력한 시맨틱 지식을 활용하되, 이를 비디오 도메인으로 확장하고 시간적 일관성을 보장하는 메커니즘을 결합하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Inflated UNet 구조 도입**: T2I 모델의 2D 컨볼루션과 self-attention을 시공간(spatio-temporal) 영역으로 확장하여 비디오 전체의 컨텍스트를 이해할 수 있도록 설계하였다.
2. **2단계 최적화 전략**: 단순한 텍스트 임베딩 최적화를 넘어, Cross-attention 레이어를 정교하게 파인튜닝하는 2단계 학습 절차를 통해 세밀한 입도의 마스크를 생성한다.
3. **Tracking-based Voting 및 Low-pass Regularization**: CoTracker를 이용한 픽셀 추적과 다수결 투표(Voting) 메커니즘을 통해 플리커(flicker) 현상을 줄이고, DCT 기반의 저역 통과 필터(Low-pass filter)를 통해 참조 이미지의 구조적 특성을 유지한다.
4. **SMITE-50 데이터셋 제안**: 다중 입도 어노테이션을 포함하여 포즈 변화, 폐색(occlusion) 등 도전적인 시나리오를 포함한 벤치마크 데이터셋을 구축하였다.

## 📎 Related Works

### 관련 연구 및 한계

- **Part-based Semantic Segmentation**: 객체의 세부 파트를 나누는 연구들이 존재하지만, 대개 특정 객체에 특화된 수동 큐레이션 정보에 의존하거나 텍스트로 묘사 가능한 파트만 분할할 수 있다는 한계가 있다.
- **Video Segmentation (VSS, VIS, VOS)**: 기존의 VOS(Video Object Segmentation) 모델들은 대개 비디오 내의 특정 프레임에 대한 어노테이션을 필요로 하며, 매우 세밀한 파트 단위의 세그멘테이션이나 학습되지 않은 데이터셋에 대한 일반화 성능이 떨어진다.
- **Video Diffusion Models**: 최근 Diffusion 모델을 이용한 비디오 생성 및 편집 연구가 활발하지만, 세그멘테이션에 적용했을 때 시간적 일관성이 부족하여 결과물이 깜빡이는(flickering) 문제가 빈번하게 발생한다.

### 차별점

SMITE는 비디오 프레임 자체의 어노테이션 없이 오직 **외부 참조 이미지**만을 사용하며, Inversion-free 모델에 포인트 추적 알고리즘과 에너지 기반 가이던스 최적화를 결합함으로써 고해상도의 세밀한 파트 분할과 시간적 일관성을 동시에 달성하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

SMITE는 사전 학습된 Stable Diffusion (SD) 모델을 기반으로 하며, 이를 비디오 처리가 가능하도록 **Inflated UNet**으로 확장한다. 전체 파이프라인은 [참조 이미지 학습 $\rightarrow$ 비디오 잠재 공간(latent space) 투영 $\rightarrow$ 추론 시 가이던스 최적화] 순으로 진행된다.

### 2. 학습 및 최적화 절차

#### (1) 학습 가능한 세그먼트 생성 (Learning Generalizable Segments)

참조 이미지 $\text{I}$와 텍스트 임베딩 $\text{T}$를 입력으로 하여 WAS(Weighted Accumulated Self-Attention) 맵 $\text{S}$를 생성한다.
$$S = \psi_{\theta}(I, T)$$
학습은 두 단계로 이루어진다.

- **Phase 1**: 모델 가중치를 동결하고 텍스트 임베딩 $\text{T}$만을 빠르게 최적화하여 초기 지점을 잡는다.
- **Phase 2**: 낮은 학습률로 Cross-attention 레이어의 가중치 $\theta$와 텍스트 임베딩을 동시에 파인튜닝하여 정교한 마스크를 생성한다.

#### (2) WAS (Weighted Accumulated Self-Attention) Map

세그멘테이션의 핵심인 WAS 맵은 Cross-attention ($\text{A}^{ca}$)과 Self-attention ($\text{A}^{sa}$)의 결합으로 정의된다.
$$S^{WAS} = \text{Sum}(\text{Flatten}(R^{ca}) \odot A^{sa})$$
여기서 $R^{ca}$는 다운샘플링된 Cross-attention 맵이다. 이는 텍스트 기반의 위치 정보와 이미지 내부의 전역적 공간 정보를 결합하여 경계선이 뚜렷한 세그먼트를 가능하게 한다.

### 3. 시간적 일관성 확보 메커니즘

#### (1) Segment Tracking and Voting

CoTracker를 이용해 픽셀의 궤적을 추적하고, 시간 윈도우 $w$ 내에서 가장 빈번하게 등장하는 라벨을 선택하는 **Temporal Voting**을 수행한다.
$$S^{Tracked}(x_t, y_t) = \text{Avg}(S(x_l, y_l) \mid \hat{Y}(x_l, y_l) = F), \forall l \in (t-\frac{w}{2}, t+\frac{w}{2})$$
여기서 $F$는 윈도우 내 최빈 라벨이다.

#### (2) Low-pass Regularization

추적 과정에서 발생할 수 있는 세그먼트의 표류(drift)를 막기 위해, 초기 디노이징 단계의 구조를 유지하도록 DCT(Discrete Cosine Transform) 기반의 저역 통과 필터를 적용한다.
$$E_{Reg} = \|\omega(S) - \omega(S^{ref})\|_1$$
여기서 $\omega$는 DCT와 저역 통과 필터 $\text{H}_l$의 곱으로 정의된다.

### 4. 시공간 가이던스 최적화 (Spatio-temporal Guidance)

최종적으로 추론 시점에 다음과 같은 통합 에너지 함수를 최소화하도록 잠재 변수 $z_t$를 업데이트한다.
$$E_{Total} = \lambda_{Tracking} \cdot E_{Tracking} + \lambda_{Reg} \cdot E_{Reg}$$
$$z'_t \leftarrow z_t - \alpha_t \cdot \nabla_{z_t} E_{Total}$$
이 과정을 통해 비디오 전체에서 플리커가 억제되고 참조 이미지의 시맨틱이 유지되는 일관된 세그멘테이션이 완성된다.

## 📊 Results

### 실험 설정

- **데이터셋**: 직접 구축한 **SMITE-50** (말, 얼굴, 자동차, Non-Text 4개 카테고리, 50개 비디오).
- **평가 지표**: mIOU (Mean Intersection Over Union), F-measure.
- **비교 대상**:
  - Baseline-I: SLiMe (프레임별 독립 적용)
  - Baseline-II: SLiMe + CoTracker (단순 픽셀 추적)
  - Baseline-III: SLiMe + Inflated UNet
  - GSAM2: 최신 제로샷 비디오 세그멘테이션 모델

### 주요 결과

- **정량적 결과**: 모든 카테고리에서 SMITE가 가장 높은 성능을 보였다. 특히 **Mean mIOU 75.14%**를 달성하며 baseline-I(66.22%) 대비 큰 성능 향상을 보였다.
- **Non-Text 카테고리**: 텍스트로 묘사하기 어려운 세그먼트의 경우, GSAM2 등 텍스트 기반 모델은 작동하지 않지만 SMITE는 참조 이미지를 통해 성공적으로 분할하였다.
- **사용자 연구**: 25명의 참여자를 대상으로 한 평가에서 세그멘테이션 품질과 모션 일관성 모두 SMITE가 가장 높은 선호도를 얻었다.
- **일반화 능력**: 말(Horse) 데이터로 학습한 모델이 기린이나 낙타와 같은 유사 구조 객체에서도 준수한 성능을 보였으며, 세단 학습 후 SUV를 분할하는 등 클래스 내 변형에도 강건함을 보였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 Diffusion 모델의 시맨틱 표현력과 고전적인 포인트 추적 및 주파수 필터링 기법을 유기적으로 결합하였다. 특히 비디오 프레임 없이 외부 이미지 몇 장만으로 유연한 입도의 분할이 가능하다는 점은 VFX 산업 등 실무 적용 가능성이 매우 높음을 시사한다.

### 한계 및 비판적 해석

1. **해상도 및 세밀함의 한계**: WAS 맵 생성 시 사용되는 Cross-attention의 저해상도 특성으로 인해 매우 얇은 선이나 아주 작은 객체를 분할하는 데 어려움이 있다 (DAVIS 데이터셋 실험에서 확인됨).
2. **추론 속도**: Diffusion 과정 내에서 에너지 기반의 최적화(backpropagation)를 반복 수행하므로, 실시간 처리와는 거리가 멀며 연산 비용이 높다.
3. **추적기 의존성**: CoTracker와 같은 외부 추적 알고리즘에 의존하므로, 추적기 자체가 실패하는 급격한 움직임이나 완전한 폐색 상황에서는 한계가 존재할 수 있다.

## 📌 TL;DR

SMITE는 사전 학습된 **T2I Diffusion 모델을 비디오로 확장(Inflated UNet)**하고, **포인트 추적 기반 투표 메커니즘**과 **저역 통과 정규화**를 도입하여, 소수의 참조 이미지로만 비디오 내 객체를 임의의 입도로 일관되게 분할하는 프레임워크이다. 이 연구는 특히 텍스트로 설명 불가능한 세부 파트의 비디오 세그멘테이션을 가능하게 함으로써, 향후 고품질 영상 편집 및 VFX 자동화 분야에 기여할 가능성이 크다.
