# Generalized Class Discovery in Instance Segmentation

Cuong Manh Hoang, Yeejin Lee, Byeongkeun Kang (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 인스턴스 분할(Instance Segmentation) 작업에서의 **Generalized Class Discovery (GCD)**이다. GCD의 목표는 레이블이 지정된 데이터($D_l$)와 레이블이 없는 데이터($D_u$)가 함께 주어졌을 때, 알려진 클래스(Known classes)뿐만 아니라 레이블 없는 데이터 속에 숨겨진 새로운 클래스(Novel classes)를 스스로 발견하고, 최종적으로 이 모든 클래스에 대해 인스턴스 분할을 수행할 수 있는 모델을 구축하는 것이다.

이 문제의 중요성은 현실 세계의 데이터가 본질적으로 **Long-tailed distribution(롱테일 분포)**을 가진다는 점에 있다. 즉, 일부 클래스는 매우 많은 인스턴스를 가지는 반면, 대다수의 클래스는 매우 적은 수의 인스턴스만을 가진다. 기존의 GCD 접근 방식들은 주로 균형 잡힌 데이터셋을 가정했기 때문에, 이러한 불균형한 분포가 존재하는 인스턴스 분할 작업에 적용했을 때 성능 저하가 발생하는 문제가 있었다. 따라서 본 논문의 목표는 데이터 불균형 문제를 해결하여 알려진 클래스와 새로운 클래스 모두에서 높은 분할 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 데이터의 불균형성을 고려하여 학습 과정의 각 단계(특징 추출 $\rightarrow$ 클래스 발견 $\rightarrow$ 최종 학습)에 맞춤형 메커니즘을 도입하는 것이다.

1.  **Instance-wise Temperature Assignment (ITA)**: Contrastive Learning 과정에서 모든 샘플에 동일한 온도 파라미터($\tau$)를 적용하는 대신, 샘플이 Head 클래스(빈도가 높은 클래스)에 속할 확률에 따라 온도를 다르게 할당한다. 이를 통해 Head 클래스는 그룹 단위의 변별력을, Tail 클래스는 개별 인스턴스 단위의 변별력을 높인다.
2.  **Reliability-based Dynamic Learning (RDL)**: 의사 레이블(Pseudo-labels)의 신뢰도를 측정할 때, 모든 클래스에 동일한 기준을 적용하면 Tail 클래스의 샘플들이 대거 배제되는 문제가 발생한다. 이를 해결하기 위해 클래스별 상대적 신뢰도 기준을 적용하고, 학습 초기에는 다양한 샘플을 수용하다가 후기에는 엄격한 신뢰도 기준을 적용하는 동적 학습 방식을 제안한다.
3.  **Soft Attention Module (SAM)**: 인스턴스 이미지 크롭 과정에서 포함될 수 있는 배경이나 인접 객체의 노이즈를 억제하고, 객체 고유의 표현(Object-specific representations)을 효과적으로 인코딩하기 위한 효율적인 어텐션 모듈을 제안한다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 바탕으로 차별점을 제시한다.

*   **Image Classification에서의 GCD**: Vaze et al. (2022) 등이 제안한 GCD는 대조 학습과 클러스터링을 통해 새로운 클래스를 발견한다. 하지만 이들은 주로 정제되고 균형 잡힌 데이터셋을 가정했다. 최근 BaCon이나 ImbaGCD 같은 연구들이 롱테일 분포를 다루기 시작했으나, 이는 이미지 분류 작업에 국한되어 있다.
*   **Segmentation 및 Detection에서의 Class Discovery**: Semantic Segmentation에서의 NCD나 Object Detection에서의 GCD 연구가 있었으며, 특히 Fomenko et al. (2022)의 RNCDL은 인스턴스 분할에서 새로운 클래스를 발견하려 시도했다.
*   **차별점**: 기존 연구들은 주로 세미-수퍼바이즈드 대조 학습과 의사 레이블 생성 방식을 그대로 차용했다. 반면, 본 논문은 인스턴스 분할 데이터의 본질적인 '불균형성'에 집중하여, 온도 파라미터의 인스턴스별 할당(ITA)과 클래스별 동적 신뢰도 기준(RDL)이라는 구체적인 해결책을 제시했다는 점에서 차별화된다.

## 🛠️ Methodology

전체 파이프라인은 크게 세 단계로 구성된다: **(1) 클래스 불가지론적 마스크 생성 $\rightarrow$ (2) GCD 모델 학습을 통한 클래스 발견 $\rightarrow$ (3) 최종 인스턴스 분할 네트워크 학습**.

### 1. Generalized Class Discovery (GCD)
먼저 GGN과 같은 모델을 사용하여 모든 객체에 대한 클래스 불가지론적(Class-agnostic) 마스크를 생성하고, 이를 통해 객체 영역만 크롭하여 이미지 세트를 구성한다.

#### Instance-wise Temperature Assignment (ITA)
데이터 불균형을 해결하기 위해, 각 인스턴스의 'Headness' 점수 $\hat{h}_i$를 임베딩 공간에서의 밀도로 추정한다.
$$\hat{h}_t^i := \frac{\sum_{z'_j \in \hat{Z}_{top K}^i} \exp(z_i^T z'_j)}{\sum_{z'_j \in \hat{Z}^i} \exp(z_i^T z'_j)}$$
이 점수에 모멘텀 업데이트를 적용하여 $h_t^i$를 구하고, 이를 기반으로 각 인스턴스 $i$에 대해 온도 $\tau_i$를 $\tau_{min}$과 $\tau_{max}$ 사이에서 선형적으로 할당한다. 수정된 대조 학습 손실 함수는 다음과 같다.
$$L_{u}^{rep} := -\log \frac{\exp(z_i^T z'_i / \tau_i)}{\sum_{z'_j \in \hat{Z}^i} \exp(z_i^T z'_j / \tau_i)}$$
이를 통해 Head 클래스는 높은 온도(그룹별 변별력)를, Tail 클래스는 낮은 온도(인스턴스별 변별력)를 갖게 된다.

#### Soft Attention Module (SAM)
객체 이미지 내의 배경 노이즈를 줄이기 위해 SAM을 CNN 백본의 각 스테이지에 통합한다. 공간 평균 풀링과 깊이 감소(Depth reduction)를 통해 효율적인 어텐션 맵 $S$를 생성하며, 최종 출력은 다음과 같이 계산된다.
$$O := S \odot F := \sigma(\nu([A, G])) \odot F$$
여기서 $A$는 풀링된 특징 맵과 원본 특징 맵 $F$ 사이의 Pairwise affinity matrix이며, $G$는 채널 차원의 글로벌 평균 풀링 맵이다. 학습 시에는 객체 마스크를 활용하여 경계 지역의 가중치를 낮춘 $\text{L}_{att}$ 손실 함수를 사용하여 최적화한다.

#### Deep Clustering
별도의 클러스터링 단계 없이 KL-divergence 기반의 손실 함수 $\text{L}_{u}^{cls}$와 지도 학습 손실 함수 $\text{L}_{s}^{cls}$를 결합하여 GCD 모델 $f_d(\cdot)$를 학습시킨다.

### 2. Reliability-Based Dynamic Learning (RDL)
GCD 모델에서 생성된 의사 레이블을 사용하여 최종 분할 네트워크 $f_s(\cdot)$를 학습시킨다. 이때 레이블의 신뢰도를 측정하기 위해 여러 체크포인트 모델들 사이의 확률 분포 안정성(Stability) $s_i$를 계산한다.
$$s_i = \sum_{\bar{t}=1}^{\bar{T}-1} \frac{1}{KL(q_{\bar{T}i} || q_{\bar{t}i})}$$
본 논문은 모든 클래스에 동일한 $r\%$ 컷오프를 적용하는 대신, **클래스 내에서의 상대적 순위**를 기준으로 신뢰도를 평가한다. 또한, 학습 진행도($t$)에 따라 신뢰도 가중치 $\kappa_t^i$를 동적으로 조정한다.
$$\kappa_t^i = s \left(1 - \left[\max\left(\frac{t - \bar{t}_i}{T_{is}}, 0\right)\right]^2\right)$$
최종 손실 함수 $\text{L}_{is}$는 레이블 데이터에 대한 손실과 의사 레이블 데이터에 $\kappa_t^i$ 가중치를 곱한 손실의 합으로 정의된다.

## 📊 Results

### 실험 설정
*   **데이터셋**: (1) $\text{COCO}_{half} + \text{LVIS}$ (80개 기지 클래스 $\rightarrow$ 1,123개 신규 클래스 발견), (2) $\text{LVIS} + \text{VG}$ (1,203개 기지 클래스 $\rightarrow$ 2,726개 신규 클래스 발견).
*   **평가 지표**: $mAP_{.50:.05:.95}$를 사용하여 전체(all), 기지(known), 신규(novel) 클래스에 대해 측정한다.
*   **비교 대상**: ORCA, UNO, SimGCD, RNCDL 및 최신 GCD 모델($\mu\text{GCD}$, $\text{NCDLR}$)을 적용한 베이스라인.

### 주요 결과
*   **정량적 성과**: $\text{COCO}_{half} + \text{LVIS}$ 설정에서 제안 방법은 $\text{mAP}_{all} 12.85$, $\text{mAP}_{novel} 11.24$를 기록하며 SOTA인 RNCDL(6.69 / 5.16)을 크게 상회하였다. $\text{LVIS} + \text{VG}$ 설정에서도 모든 지표에서 가장 높은 성능을 보였다.
*   **Ablation Study**: ITA, SAM, RDL 각각의 모듈을 추가할 때마다 성능이 단계적으로 향상됨을 확인하였다. 특히 RDL이 없을 때보다 있을 때 $\text{mAP}_{all}$이 $10.71 \rightarrow 12.85$로 크게 상승하여, 불균형 데이터에서 동적 신뢰도 기준의 중요성을 입증했다.
*   **ITA vs TS**: 기존의 Temperature Scheduling(TS) 방식보다 본 논문의 ITA 방식이 롱테일 분포에서 더 우수한 임베딩 공간을 형성함을 확인하였다.

## 🧠 Insights & Discussion

### 강점
본 연구는 인스턴스 분할이라는 복잡한 작업에서 GCD를 수행할 때 발생하는 데이터 불균형 문제를 매우 구체적인 관점에서 접근했다. 단순히 알고리즘을 적용하는 것이 아니라, 온도 파라미터($\tau$)와 신뢰도 임계값($r\%$)이라는 하이퍼파라미터를 **인스턴스 및 클래스 수준에서 동적으로 적응**시킨 점이 성능 향상의 핵심 요인으로 분석된다.

### 한계 및 비판적 해석
1.  **데이터 가용성 가정**: 모든 레이블 및 언레이블 데이터가 학습 시작 시점에 이미 준비되어 있다고 가정한다. 하지만 실제 로봇 내비게이션과 같은 환경에서는 데이터가 순차적으로 들어오므로, 이러한 정적 학습 방식은 실시간 적용에 한계가 있다.
2.  **클래스 개수 사전 지식**: 대부분의 GCD 연구와 마찬가지로, 전체 클래스 개수에 대한 사전 지식이 필요하다는 점이 한계로 지적된다. 완전히 개방된 환경(Open-world)에서는 전체 클래스 수를 알 수 없으므로 이에 대한 추가 연구가 필요하다.
3.  **계산 복잡도**: SAM 모듈을 백본의 모든 스테이지에 추가하고, RDL을 위해 여러 체크포인트 모델을 저장하고 KL-divergence를 계산하는 과정이 추가되므로 학습 시간이 증가할 가능성이 크다.

## 📌 TL;DR

본 논문은 인스턴스 분할 작업에서 알려지지 않은 새로운 클래스를 발견하는 **Generalized Class Discovery (GCD)** 문제를 다룬다. 특히 현실 세계의 **롱테일 분포(데이터 불균형)** 문제를 해결하기 위해 $\text{(1) 인스턴스별 온도 할당(ITA), (2) 클래스별 동적 신뢰도 학습(RDL), (3) 효율적인 소프트 어텐션 모듈(SAM)}$을 제안하였다. 실험 결과, 제안 방법은 기존 SOTA 모델들을 크게 앞지르며, 특히 새로운 클래스(Novel classes)의 분할 성능을 획기적으로 향상시켰다. 이는 향후 오픈월드 인스턴스 분할 시스템 구축에 중요한 기반 기술이 될 것으로 기대된다.