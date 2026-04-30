# Universal Domain Adaptation for Semantic Segmentation

Seun-An Choe, Keon-Hee Park, Jinwoo Choi, Gyeong-Moon Park (2025)

## 🧩 Problem to Solve

본 논문은 시맨틱 세그멘테이션을 위한 비지도 도메인 적응(Unsupervised Domain Adaptation for Semantic Segmentation, UDA-SS)에서 발생하는 현실적인 제약 문제를 해결하고자 한다. 기존의 UDA-SS 방법론들은 소스(Source) 도메인과 타겟(Target) 도메인 간의 클래스 구성(Category settings)이 미리 알려져 있다는 가정을 전제로 한다. 그러나 실제 환경에서는 타겟 도메인의 레이블을 알 수 없으므로, 소스에는 없지만 타겟에만 존재하는 클래스(Target-private classes)나 타겟에는 없지만 소스에만 존재하는 클래스(Source-private classes)가 빈번하게 발생한다.

이러한 클래스 불일치는 소스-프라이빗 클래스가 타겟 도메인의 클래스와 잘못 정렬되는 negative transfer 현상을 야기하며, 결과적으로 모델의 성능을 심각하게 저하시킨다. 특히 타겟-프라이빗 클래스가 존재할 때, 공통 클래스(Common classes)의 pseudo-label 신뢰도(confidence score)가 낮아져 이들이 '알 수 없음(unknown)'으로 오분류되는 문제가 발생한다. 따라서 본 논문의 목표는 클래스 구성에 대한 사전 지식 없이도 강건하게 적응할 수 있는 Universal Domain Adaptation for Semantic Segmentation (UniDA-SS) 시나리오를 정의하고, 이를 해결하기 위한 UniMAP 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 공통 클래스에 대한 신뢰도 점수를 높여 소스-프라이빗 클래스와의 혼동을 줄이는 것이다. 이를 위해 다음의 두 가지 핵심 설계를 제안한다.

1. **Domain-Specific Prototype-based Distinction (DSPD)**: 공통 클래스를 단순히 하나의 클래스로 처리하지 않고, 소스와 타겟 각각을 위한 두 개의 도메인 특화 프로토타입(Domain-specific prototype)을 할당한다. 이를 통해 도메인 간의 특징 차이를 수용하면서도, 두 프로토타입 모두에 가깝게 위치하는 픽셀을 공통 클래스로 식별하여 신뢰도를 높인다.
2. **Target-based Image Matching (TIM)**: 학습 과정에서 소스-프라이빗 클래스로 인해 공통 클래스 학습이 소홀해지는 것을 방지하기 위해, 타겟 pseudo-label의 분포를 기반으로 공통 클래스 픽셀이 많이 포함된 소스 이미지를 우선적으로 선택하여 배치를 구성함으로써 도메인 불변 표현 학습을 촉진한다.

## 📎 Related Works

기존의 시맨틱 세그멘테이션 UDA 연구는 크게 적대적 학습(Adversarial learning) 기반 방법과 자가 학습(Self-training) 기반 방법으로 나뉜다. 적대적 학습은 도메인 분류기를 통해 도메인 불변 특징을 학습하고, 자가 학습은 타겟 도메인의 pseudo-label을 생성하여 모델을 재학습시킨다. 하지만 이들은 모두 클래스 오버랩에 대한 사전 지식이 필요하다는 한계가 있다.

분류(Classification) 작업에서는 이미 UniDA 연구가 진행되어, 알려진 클래스에 대한 신뢰도 점수를 계산하고 낮은 점수의 샘플을 unknown으로 처리하는 방식(예: CMU, ROS, DANCE, OVANet)이 제안되었다. 그러나 시맨틱 세그멘테이션은 픽셀 단위의 정밀한 분류가 필요하여 시각적 이해 수준이 더 높아야 하므로, 분류 작업의 UniDA를 그대로 적용하기 어렵다. 본 논문은 이러한 간극을 메우기 위해 세그멘테이션 환경에 최적화된 UniDA-SS 접근 방식을 제시한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 베이스라인
본 논문은 BUS 모델의 구조를 기반으로 하며, 분류 헤드에 'unknown' 클래스를 위한 노드를 추가하여 총 $C_s + 1$개의 헤드를 구성한다. 교사 네트워크(Teacher network) $g_\phi$는 지수 이동 평균(EMA)을 통해 업데이트되며, 다음과 같은 기준으로 타겟 pseudo-label $\hat{y}_{tp}^{(j)}$를 생성한다.

$$\hat{y}_{tp}^{(j)} = \begin{cases} c', & \text{if } \max_{c'} g_\phi(x_t)^{(j, c')} \ge \tau_p \\ C_s+1, & \text{otherwise} \end{cases}$$

여기서 $\tau_p$는 unknown 클래스를 할당하기 위한 고정 임계값이다.

### 2. Domain-Specific Prototype-based Distinction (DSPD)
DSPD는 픽셀 임베딩 공간에서 클래스별로 두 개의 프로토타입(소스용 $p_{c_s}$, 타겟용 $p_{c_t}$)을 정의한다. 프로토타입 간의 안정적인 거리를 유지하기 위해 고정된 Simplex Equiangular Tight Frame (ETF) 구조를 사용한다.

**학습 목표 및 손실 함수:**
픽셀 임베딩 $i$를 프로토타입에 정렬시키기 위해 세 가지 손실 함수를 결합한 $L_{proto}$를 사용한다.
- **Cross Entropy Loss ($L_{CE}$)**: 픽셀을 해당 클래스의 프로토타입에 가깝게, 나머지 프로토타입과는 멀게 만든다.
- **Pixel-Prototype Contrastive Learning ($L_{PPC}$)**: 전체 공간에서 해당 클래스 프로토타입과의 유사도를 높인다.
- **Pixel-Prototype Distance Optimization ($L_{PPD}$)**: 픽셀 임베딩과 프로토타입 간의 거리를 직접적으로 최적화한다.

$$L_{proto} = L_{CE} + \lambda_1 L_{PPC} + \lambda_2 L_{PPD}$$

**Prototype-based Weight Scaling:**
공통 클래스 픽셀은 소스와 타겟 프로토타입 모두에 유사한 거리를 가지는 반면, 프라이빗 클래스는 어느 한쪽으로 치우치는 경향이 있다. 이를 이용하여 픽셀별 가중치 $w$를 계산한다. $d_s, d_t$를 각각 소스와 타겟 프로토타입과의 코사인 유사도라고 할 때:

$$w = \frac{2(d_s+1)(d_t+1)}{(d_s+1) + (d_t+1)}$$

이 $w$는 타겟 pseudo-label 생성 시의 임계값 판단과 타겟 손실 함수 $L_{t\_seg}$에 곱해져, 공통 클래스일 확률이 높은 픽셀의 학습 비중을 높인다.

### 3. Target-based Image Matching (TIM)
TIM은 타겟 도메인에서 부족한 공통 클래스의 학습량을 확보하기 위해 소스 이미지를 전략적으로 샘플링한다.

1. **클래스 분포 계산**: 타겟 pseudo-label에서 각 클래스의 비율 $f_c$를 계산하고, 희귀 클래스에 더 높은 가중치를 주는 $\hat{f}_c = \text{softmax}((1-f_c)/T)$를 구한다.
2. **이미지 매칭 스코어 계산**: 각 소스 이미지 $x_s$에 대해, 타겟과 겹치는 공통 클래스 픽셀 수와 가중치의 곱인 $S_s$를 계산한다.
   $$S_s = \sum_{c \in c^*} n_{s_c} \hat{f}_c$$
3. **배치 구성**: $S_s$가 가장 높은 소스 이미지를 선택하여 타겟 이미지와 한 배치에 묶어 학습시킨다. 이를 통해 공통 클래스에 대한 도메인 불변 표현 학습을 극대화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Pascal-Context $\rightarrow$ Cityscapes (Real-to-Real), GTA5 $\rightarrow$ IDD (Synthetic-to-Real)의 OPDA-SS(Open Partial Domain Adaptation) 벤치마크를 사용한다.
- **평가 지표**: 공통 클래스의 mIoU와 타겟-프라이빗 클래스의 IoU의 조화 평균인 **H-Score**를 사용하여 종합적인 성능을 측정한다.
- **비교 대상**: UAN, UniOT, MLNet, DAFormer, HRDA, MIC, BUS 등 최신 UDA-SS 및 UniDA 방법론.

### 주요 결과
- **Pascal-Context $\rightarrow$ Cityscapes**: UniMAP은 Common(60.94), Private(31.27), H-Score(41.33)를 기록하며 baseline 및 기존 SOTA 모델들을 크게 상회하였다. 특히 H-Score 기준 BUS 대비 약 11.22 포인트 향상되었다.
- **GTA5 $\rightarrow$ IDD**: 공통 클래스 성능은 BUS보다 약간 낮았으나, Private 클래스 성능(34.78)과 H-Score(45.51)에서 압도적인 우위를 보였다.
- **정성적 결과**: 시각화 결과, 타 모델들이 공통 클래스를 target-private으로 오분류하거나 그 반대의 경우를 보이는 것과 달리, UniMAP은 "sidewalk"와 같은 공통 클래스와 target-private 영역을 모두 정확하게 구분해냈다.

### 절제 실험 (Ablation Study)
- **DSPD와 TIM의 효과**: DSPD는 도메인 특화 특징을 캡처하여 성능을 높이고, TIM은 도메인 불변 표현 학습을 도와 성능을 향상시킨다. 두 모듈을 함께 사용했을 때 H-Score가 41.33으로 가장 높았다.
- **DSPD 내부 구성**: $L_{proto}$만 사용했을 때보다 가중치 $w$를 함께 사용했을 때 성능이 크게 향상되었으며, $w$는 $L_{proto}$의 가이드 없이는 효과가 낮음을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 UniDA-SS라는 새로운 시나리오를 정의하고, 이를 위해 프로토타입 기반의 특징 분리와 타겟 기반 이미지 매칭이라는 두 가지 전략을 통해 성공적으로 해결하였다.

**강점:**
- 단순히 클래스를 통합하여 학습하는 대신, 도메인별 프로토타입을 두어 도메인 간 변이를 수용하면서도 공통 클래스를 식별해내는 방식이 매우 효과적이다.
- 데이터 불균형 문제를 해결하기 위해 타겟 pseudo-label의 분포를 소스 이미지 샘플링에 반영한 TIM 전략은 실질적인 학습 효율을 높였다.

**한계 및 논의:**
- 제안된 방법론이 다양한 카테고리 시프트 설정(CDA, ODA, PDA, OPDA)에서 강건함을 보였으나, Closed-set이나 Open-set 전용 모델보다는 해당 특정 시나리오에서 성능이 약간 낮을 수 있다. 이는 범용성(Universality)을 추구하면서 발생하는 일반적인 트레이드-오프(Trade-off)로 해석된다.
- 또한, ETF 공간을 사용한 고정 프로토타입 방식이 모든 데이터셋에서 최적의 거리를 보장하는지에 대한 추가적인 분석이 필요할 수 있다.

## 📌 TL;DR

본 논문은 소스와 타겟 도메인 간의 클래스 구성이 일치하지 않는 상황에서도 작동하는 **Universal Domain Adaptation for Semantic Segmentation (UniDA-SS)**를 제안한다. 제안된 **UniMAP** 프레임워크는 도메인별 프로토타입을 통해 공통 클래스와 프라이빗 클래스를 구분하는 **DSPD**와, 타겟 분포에 맞춰 소스 데이터를 샘플링하는 **TIM**을 통해 성능을 극대화한다. 실험을 통해 기존 SOTA 모델들보다 특히 Open Partial DA 시나리오에서 월등한 성능을 입증하였으며, 이는 실제 환경의 불확실한 클래스 구성 문제를 해결하는 데 중요한 기여를 할 것으로 보인다.