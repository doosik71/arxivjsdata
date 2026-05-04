# Manta: Enhancing Mamba for Few-Shot Action Recognition of Long Sub-Sequence

Wenbo Huang, Jinghui Zhang, Guang Li, Lei Zhang, Shuoyuan Wang, Fang Dong, Jiahui Jin, Takahiro Ogawa, Miki Haseyama (2025)

## 🧩 Problem to Solve

본 논문은 Few-Shot Action Recognition (FSAR) 분야에서 긴 서브 시퀀스(long sub-sequences)를 활용할 때 발생하는 문제들을 해결하고자 한다. 일반적으로 긴 비디오 서브 시퀀스는 동작의 전체 과정을 더 효과적으로 표현할 수 있어 인식 성능 향상에 유리하지만, 다음과 같은 세 가지 주요 난관이 존재한다.

첫째, 기존의 Transformer 기반 방법론들은 높은 연산 복잡도로 인해 보통 8프레임 정도의 짧은 서브 시퀀스만 처리할 수 있다는 한계가 있다.
둘째, 최근 효율적인 롱 시퀀스 모델링으로 주목받는 Mamba 모델을 FSAR에 직접 적용할 경우, 전역적 특징(global feature) 모델링에 치중하여 동작 인식에 필수적인 국소적 특징(local feature) 모델링과 시간적 정렬(temporal alignment)을 간과하게 된다.
셋째, 서브 시퀀스의 길이가 길어질수록 동일 클래스 내에서도 촬영 조건이나 배경 등의 차이로 인한 클래스 내 분산(intra-class variance)이 누적되어, 샘플들을 효과적으로 클러스터링하는 것이 어려워진다.

따라서 본 연구의 목표는 Mamba의 효율성을 유지하면서도 국소 특징 추출, 시간적 정렬, 그리고 클래스 내 분산 문제를 해결하여 긴 서브 시퀀스 기반의 FSAR 성능을 극대화하는 Manta 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Matryoshka Mamba** 구조를 통한 국소 특징의 계층적 모델링과 **하이브리드 대조 학습(Hybrid Contrastive Learning)**을 통한 클래스 내 분산 억제이다.

1. **Matryoshka Mamba 설계**: 러시아 인형 마트료시카처럼 Inner Module이 Outer Module 내부에 중첩된 구조를 설계하였다. Inner Module은 긴 시퀀스를 조각내어 국소 특징을 강화하고, Outer Module은 이들 간의 시간적 의존성을 캡처하여 암시적인 시간적 정렬을 수행한다.
2. **멀티 스케일 모델링**: 단일 스케일이 아닌 여러 스케일($O$)의 프레임 수를 적용하여 다양한 수준의 국소 특징을 포착하고, 학습 가능한 가중치를 통해 이를 적응적으로 융합한다.
3. **하이브리드 대조 학습 패러다임**: 지도 학습 기반의 대조 학습(지원 세트 대상)과 비지도 학습 기반의 대조 학습(쿼리 세트 및 전체 샘플 대상)을 결합하여, 긴 시퀀스에서 발생하는 클래스 내 분산의 부정적 영향을 완화한다.

## 📎 Related Works

### Few-Shot Action Recognition (FSAR)

기존 FSAR의 주류는 Metric-based meta-learning으로, Transformer를 사용하여 시간적 정렬을 수행한다. OTAM은 Dynamic Time Warping (DTW)을 사용하며, TRX나 HyRSM 등은 국소 특징 모델링을 통해 성능을 높였다. 하지만 이러한 방법들은 Transformer의 연산 복잡도로 인해 매우 짧은 서브 시퀀스에만 국한되어 사용되었다는 한계가 있다.

### Mamba Architecture

SSM (State Space Models)을 기반으로 하는 Mamba는 Transformer 대비 효율적인 롱 시퀀스 모델링이 가능하다. ViM이나 VMamba와 같은 모델들이 비전 분야에 적용되었으나, FSAR에서 필수적인 국소 특징 정렬 문제는 해결하지 못했다.

### Contrastive Learning

비지도 학습을 통해 일반적인 표현력을 학습하는 대조 학습은 FSAR에서 보조 손실 함수로 사용되어 클래스 내 분산을 줄이는 데 효과적임이 알려져 있다. Manta는 이를 확장하여 지도/비지도 방식을 혼합한 하이브리드 방식을 제안한다.

## 🛠️ Methodology

### 전체 시스템 구조

Manta는 크게 **Mamba Branch**와 **Contrastive Branch**라는 두 개의 병렬 브랜치로 구성된다. 먼저 Backbone 네트워크를 통해 특징을 추출한 후, 두 브랜치를 거쳐 최종적으로 Cross-entropy loss와 Contrastive loss의 가중 합으로 학습된다.

### Matryoshka Mamba (Mamba Branch)

이 모듈은 국소 특징 강화와 시간적 정렬을 목표로 하며, Mamba-2를 기반으로 설계되었다.

1. **Inner Module (IM)**: 입력 시퀀스를 겹치지 않는 조각(fragments)으로 나누어 처리한다. 순방향($IM_{Fw}$)과 역방향($IM_{Bw}$) 두 서브 브랜치를 통해 국소 특징을 강화하며, 결과는 다음과 같이 결합된다.
   $$IM(\cdot) = \text{Linear}[IM_{Fw}(\cdot) \oplus IM_{Bw}(\cdot)]$$
2. **Outer Module (OM)**: Inner Module에서 처리된 특징들을 입력으로 받아 전체 시퀀스를 양방향으로 스캔하며 시간적 의존성을 파악한다. 이를 통해 암시적인 시간적 정렬을 수행한다.
3. **멀티 스케일 융합**: 하이퍼파라미터 $O$에 정의된 다양한 스케일 $o$에 대해 위 과정을 반복한다. 각 스케일의 중요도를 결정하기 위해 Conv2D Block과 Sigmoid를 이용한 학습 가능한 가중치 $w_o$를 계산한다.
   $$w^S_o = \text{Sigmoid}(\text{CB}(\tilde{S}^{ck}_{fo}) \oplus S^{ck}_{fi})$$
   최종 출력 $\hat{S}^{ck}_f$는 모든 스케일 결과의 평균으로 산출된다.

### 프로토타입 구성 및 거리 계산

지원 세트의 특징들을 평균하여 클래스 프로토타입 $\hat{P}_c$를 생성한다. 이후 쿼리 샘플 $\hat{Q}^r_f$와의 거리를 계산하는데, 대칭적 정렬을 위해 원본과 반전(inversion)된 텐서 간의 네 가지 거리 조합($D_1, D_2, D_3, D_4$)을 사용하며, 이들의 평균값 $D$를 통해 클래스를 예측한다.

### 하이브리드 대조 학습 (Contrastive Branch)

클래스 내 분산을 줄이기 위해 다음과 같은 하이브리드 손실 함수 $L_{hc}$를 사용한다.
$$L_{hc} = L^{con}_S + L^{con}_Q + L^{con}_{SQ}$$

- $L^{con}_S$: 라벨이 있는 지원 세트를 이용한 지도 대조 학습.
- $L^{con}_Q$: 라벨이 없는 쿼리 세트를 이용한 비지도 대조 학습.
- $L^{con}_{SQ}$: 지원 세트와 쿼리 세트를 모두 포함한 전체 샘플 대상의 비지도 학습.

### 학습 목표

최종 손실 함수는 다음과 같이 정의된다.
$$L_{total} = \lambda L_{ce} + L_{hc}$$
여기서 $L_{ce}$는 분류를 위한 Cross-entropy loss이며, $\lambda$는 두 손실 간의 균형을 맞추는 가중치 계수이다.

## 📊 Results

### 실험 설정

- **데이터셋**: SSv2, Kinetics, UCF101, HMDB51.
- **백본**: ResNet-50, ViT-B, VMamba-B.
- **설정**: 5-way 1-shot 및 5-shot 설정. 서브 시퀀스 길이 $F$는 8에서 128까지 다양하게 조정.

### 주요 결과

1. **SOTA 달성**: Manta는 모든 벤치마크 데이터셋에서 기존 방법론(TRX, MoLo 등) 및 최신 멀티모달 방법론(AMFAR)보다 높은 정확도를 기록하였다. 특히 ResNet-50 백본 기준 SSv2 1-shot에서 63.4%를 기록하며 기존 SOTA를 경신하였다.
2. **롱 시퀀스 처리 능력**: 서브 시퀀스 길이 $F$를 8에서 128까지 늘렸을 때, 기존 Transformer 기반 모델들은 메모리 부족(OOM)으로 작동하지 않았으나, Manta는 $F=128$에서도 안정적으로 동작하며 성능이 향상되는 경향을 보였다.
3. **강건성(Robustness)**: 프레임 레벨 노이즈, 샘플 레벨 노이즈, 배경 노이즈(가우시안, 비, 조명 변화)가 추가된 상황에서도 타 모델 대비 성능 하락 폭이 매우 적어 높은 강건성을 입증하였다.
4. **추론 속도**: Mamba의 효율적인 아키텍처 덕분에 Transformer 기반 모델들보다 추론 속도가 현저히 빨랐다 (예: SSv2 1-shot 기준 Manta는 4.25시간, MoLo는 7.83시간).

## 🧠 Insights & Discussion

본 논문은 FSAR에서 긴 시퀀스를 사용할 때 발생하는 **'국소 특징 소실'**과 **'분산 누적'**이라는 두 가지 핵심 문제를 구조적(Matryoshka Mamba) 및 학습적(Hybrid Contrastive Learning) 방법으로 동시에 해결하였다.

특히 주목할 점은 시간적 정렬의 학습 과정이다. DTW 스코어 분석 결과, Manta는 학습이 진행됨에 따라 큰 스케일(coarse-grained)에서 작은 스케일(fine-grained) 순으로 정렬 능력을 획득함을 확인하였다. 이는 멀티 스케일 설계가 단순히 성능을 높이는 것을 넘어, 모델이 단계적으로 정교한 시간적 특징을 학습하게 유도함을 시사한다.

한계점으로는 하이퍼파라미터 $\lambda$나 $\tau$ (temperature) 등의 설정에 따라 성능 변동이 존재한다는 점이 있으나, 광범위한 실험을 통해 최적값을 제시하였다. 또한, 매우 긴 시퀀스($F>128$)에서의 확장성나 더 복잡한 환경에서의 일반화 성능에 대한 추가 연구가 필요할 것으로 보인다.

## 📌 TL;DR

Manta는 긴 비디오 서브 시퀀스를 활용하는 Few-Shot Action Recognition을 위해 제안된 프레임워크이다. 국소 특징 모델링과 시간적 정렬을 수행하는 **Matryoshka Mamba** 구조와, 클래스 내 분산을 억제하는 **하이브리드 대조 학습**을 결합하여, 연산 효율성을 유지하면서도 SOTA 성능을 달성하였다. 특히 기존 모델들이 처리하지 못하는 매우 긴 시퀀스에서도 안정적인 성능과 빠른 추론 속도를 보여, 향후 실시간 비디오 이해 및 소량 데이터 기반 동작 인식 연구에 중요한 기여를 할 것으로 기대된다.
