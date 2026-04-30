# FedDiverse: Tackling Data Heterogeneity in Federated Learning with Diversity-Driven Client Selection

Gergely D. Németh, Eros Fan, Yeat Jeng Ng, Barbara Caputo, Miguel Ángel Lozano, Nuria Oliver, Novi Quadrianto (2025)

## 🧩 Problem to Solve

본 논문은 연합 학습(Federated Learning, FL) 환경에서 발생하는 **통계적 데이터 이질성(Statistical Data Heterogeneity)** 문제를 해결하고자 한다. 실제 환경의 FL에서는 클라이언트들이 보유한 데이터가 독립 동일 분포(IID)를 따르지 않으며, 심하게 불균형한 경우가 많다. 이러한 이질성은 서버 모델의 일반화 능력을 저하시키고, 수렴 속도를 늦추며, 전반적인 성능 하락을 초래한다.

특히, 저자들은 기존의 연구들이 주로 클래스 불균형(Class Imbalance)에만 집중했음을 지적하며, 보다 세밀한 관점에서 다음과 같은 세 가지 유형의 데이터 이질성을 정의하고 이를 해결하는 것을 목표로 한다:
1. **클래스 불균형 (Class Imbalance, CI):** 타겟 라벨의 분포가 학습 세트와 테스트 세트 간에 서로 다른 경우이다.
2. **속성 불균형 (Attribute Imbalance, AI):** 특정 속성(Attribute)의 발생 확률이 학습 데이터에서 매우 낮아, 모델이 다수 속성에 편향되는 경우이다.
3. **가짜 상관관계 (Spurious Correlation, SC):** 판별에 불필요한 속성(예: 배경)과 타겟 라벨 사이에 통계적 의존성이 존재하여, 모델이 본질적인 특징이 아닌 가짜 특징을 학습하는 경우이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **데이터 분포가 서로 보완적인(Complementary) 클라이언트들을 전략적으로 선택하여 학습에 참여시킴으로써, 모델이 다양한 데이터 패턴에 노출되게 하여 일반화 성능과 강건성(Robustness)을 높이는 것**이다.

주요 기여 사항은 다음과 같다:
- **데이터 이질성 분석 프레임워크:** CI, AI, SC를 측정하기 위한 6가지 지표(전역 3개, 클라이언트 3개)를 제안하여 데이터 이질성을 정밀하게 정의하였다.
- **벤치마크 데이터셋 구축:** 다양한 수준의 CI, AI, SC를 포함하는 7개의 컴퓨터 비전 데이터셋을 생성하고 공유하여 실제 세계의 이질적인 상황을 시뮬레이션하였다.
- **FEDDIVERSE 알고리즘 제안:** 각 클라이언트의 데이터 이질성 특성을 추정하고, 이를 바탕으로 서로 다른 이질성 프로필을 가진 클라이언트들을 선택하는 새로운 클라이언트 선택 알고리즘을 개발하였다.

## 📎 Related Works

### 기존 연구 및 한계
- **FL 데이터 이질성 해결:** $\text{FedProx}$나 $\text{FedDyn}$과 같은 정규화 기반 방법, $\text{MOON}$과 같은 분산 감소 방법, $\text{FedAvgM}$과 같은 서버 측 최적화 방법 등이 제안되었다. 그러나 이들은 대부분 클래스 불균형 위주로 접근하며, 특히 **가짜 상관관계(SC) 문제**를 명시적으로 다루지 않는다.
- **중앙 집중식 ML의 SC 해결:** $\text{LFF}$나 $\text{Just-Train-Twice}$와 같이 편향된 모델을 먼저 학습시킨 후 이를 이용해 디바이아싱(De-biasing)하는 기법들이 존재한다. 하지만 이러한 방법들을 FL의 프라이버시 제약 조건 하에서 적용하는 것은 어렵다.
- **클라이언트 선택 및 가중치 조절:** $\text{POW-D}$나 $\text{FedPNS}$처럼 손실 값이나 그래디언트 유사도를 기반으로 클라이언트를 선택하는 방법들이 있다. 하지만 이들 역시 속성 불균형이나 가짜 상관관계와 같은 세밀한 통계적 특성을 고려하지 않는다.

### 차별점
$\text{FEDDIVERSE}$는 단순한 성능 지표나 그래디언트가 아닌, 데이터의 통계적 분포 특성(CI, AI, SC)을 기반으로 클라이언트를 선택한다. 특히, 서로 보완적인 특성을 가진 클라이언트들을 조합하여 단일 전역 모델이 모든 분포에 대해 강건한 성능을 갖도록 유도한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 데이터 이질성 측정 지표
데이터셋 $D$에 대해 엔트로피 $H$와 상호 정보량 $I$를 사용하여 다음과 같이 측정한다.
- 클래스 불균형: $\Delta_{CI}(D) = 1 - H(Y)/\log|Y|$
- 속성 불균형: $\Delta_{AI}(D) = 1 - H(A)/\log|A|$
- 가짜 상관관계: $\Delta_{SC}(D) = 2I(Y;A)/(H(Y) + H(A))$

전역 지표(GCI, GAI, GSC)는 전체 데이터의 합집합에서 계산하며, 클라이언트 지표(CCI, CAI, CSC)는 각 클라이언트 $\text{k}$의 지표 $\Delta(D_k)$ 값들의 평균으로 계산한다.

### 2. 데이터 이질성 추정 (Estimation of Interaction Matrices)
클라이언트는 자신의 속성 라벨을 정확히 알 수 없으므로, 다음의 3단계 과정을 통해 상호작용 행렬 $\tilde{N}_k$와 데이터 이질성 트리플렛(Data Heterogeneity Triplet, DHT) $\tilde{\Delta}_k = [\Delta_{CI}, \Delta_{AI}, \Delta_{SC}]^\top$을 추정한다. 이 과정은 학습 시작 전 한 번만 수행된다.

1. **사전 학습 (Pre-training):** $\text{FedAvg}$를 사용하여 짧은 라운드 동안 전역 모델 $\theta^{T_0}$를 학습시킨다.
2. **편향 모델 학습 (Learning a Biased Model):** 각 클라이언트는 $\theta^{T_0}$를 기반으로 $\text{Generalized Cross Entropy (GCE)}$ 손실 함수 $\ell_{GCE}$를 사용하여 로컬 모델 $\bar{f}_k$를 오버피팅시킨다. GCE 손실은 모델이 가짜 상관관계와 같은 '학습하기 쉬운' 패턴에 의존하게 만들어, 다수 그룹 $\tilde{G}_k$와 소수 그룹 $\tilde{g}_k$를 구분할 수 있게 한다.
3. **속성 분류기 학습 (Attribute Classifier):** 다수/소수 그룹의 크기 차이가 가장 작은 '피벗 클래스(Pivot Class)' $\hat{y}$를 선정하고, 해당 클래스의 샘플들 $\hat{D}_k$를 사용하여 속성 분류기 $\hat{\psi}$를 학습시킨다. 이를 통해 추정된 상호작용 행렬 $\tilde{N}_k$를 얻고, 최종적으로 $\tilde{\Delta}_k$를 계산하여 서버에 전송한다.

### 3. FEDDIVERSE 클라이언트 선택 절차
서버는 수집된 DHT를 바탕으로 매 라운드 다음과 같은 순서로 클라이언트를 선택한다.

1. **확률적 선택 (SC 중심):** 가짜 상관관계 성분 $\tilde{\Delta}_3$을 기반으로 확률 분포 $p^{SC}$를 생성하여 첫 번째 클라이언트 $k_p$를 선택한다.
   $$p^{SC} = \frac{\tilde{\Delta}_3}{\|\tilde{\Delta}_3\|_1}$$
2. **보완적 선택 (AI 또는 CI 중심):** 선택된 $k_p$의 정규화된 DHT와 내적(Dot Product) 값이 가장 작은, 즉 가장 덜 정렬된(Least Aligned) 클라이언트 $k_c$를 선택한다.
   $$k_c = \arg \min_{k \in K \setminus \{k_p\}} \langle \tilde{\Delta}_{k_p}, \tilde{\Delta}_k \rangle$$
3. **직교적 선택 (CI 또는 AI 중심):** 앞서 선택된 두 클라이언트의 DHT 벡터들의 외적(Cross Product) 결과와 가장 정렬이 잘 된 클라이언트 $k_r$을 선택하여, 남은 차원의 이질성을 보완한다.
   $$k_r = \arg \max_{k \in K \setminus \{k_p, k_c\}} \langle \tilde{\Delta}_{k_p} \times \tilde{\Delta}_{k_c}, \tilde{\Delta}_k \rangle$$

선택 순서(SC $\rightarrow$ CI $\rightarrow$ AI)는 매 3명의 클라이언트를 선택할 때마다 순환(Rotate)되어 다양성을 확보한다.

## 📊 Results

### 실험 설정
- **데이터셋:** WaterBirds, Spawrious(5종: GSC, GCI, GAI, 4-class, GCI-100), CMNIST 총 7종.
- **모델 및 최적화:** MobileNet v2 (Group Normalization 적용), $\text{FedAvgM}$을 기본 최적화 알고리즘으로 사용.
- **평가 지표:** **Worst-group accuracy** (가장 성능이 낮은 그룹의 정확도)를 측정하여 모델의 강건성을 평가한다.

### 주요 결과
- **성능 우위:** $\text{FEDDIVERSE}$는 모든 데이터셋에서 베이스라인($\text{Uniform random}$, $\text{Round robin}$, $\text{FedNova}$, $\text{POW-D}$, $\text{FedPNS}$, $\text{HCSFED}$) 대비 최상위 또는 차상위 성능을 기록하였다. (표 I 참조)
- **최적화 알고리즘과의 결합:** $\text{FedAvg}$, $\text{FedProx}$, $\text{FedAvgM}$ 등 다양한 최적화 기법과 결합했을 때 모두 랜덤 선택보다 성능이 향상되었으며, 특히 $\text{FedAvgM}$과 결합했을 때 가장 뛰어난 성능을 보였다. (표 III 참조)
- **오버헤드 분석:** 클라이언트 측 계산 비용은 초기 1회 추정 단계에서만 발생하며, 통신 비용은 클라이언트당 단 3개의 스칼라 값(DHT)을 전송하는 수준으로 매우 낮다.

### Ablation Study
- **추정 vs 실제 값:** 실제 상호작용 행렬 $N_k$를 알고 있는 이상적인 경우보다 성능이 약간 낮았으나, 추정된 $\tilde{\Delta}_k$를 사용하는 방식이 프라이버시를 보호하면서도 충분히 경쟁력 있는 성능을 낸다는 것을 확인하였다. (표 IV 참조)

## 🧠 Insights & Discussion

### 강점
- **가짜 상관관계 해결:** 기존 FL 연구들이 간과했던 SC 문제를 정밀한 지표 정의와 클라이언트 선택 전략을 통해 해결하였다.
- **프라이버시 및 효율성:** 원본 데이터나 상세 분포를 공유하지 않고 단 3개의 통계치(DHT)만 공유함으로써 프라이버시를 유지하며, 통신 오버헤드를 최소화하였다.
- **범용성:** 특정 FL 최적화 알고리즘에 종속되지 않고, 클라이언트 선택 단계에서 독립적으로 작동하므로 기존의 다양한 FL 프레임워크에 즉시 적용 가능하다.

### 한계 및 논의사항
- **속성 수의 제한:** 본 논문은 속성의 수가 2개($|A|=2$)인 상황을 가정하고 설계되었다. 속성이 매우 많은 복잡한 데이터셋으로 확장할 경우, DHT의 차원 확장과 직교 선택 전략의 수정이 필요할 것으로 보인다.
- **사전 학습 의존성:** $\text{FEDDIVERSE}$의 핵심인 DHT 추정을 위해 사전 학습 단계($T_0$)가 필수적이다. 이 단계의 라운드 수나 초기 모델의 품질이 추정 정확도에 영향을 줄 수 있다.

## 📌 TL;DR

본 논문은 연합 학습에서 모델 성능을 저하시키는 **클래스 불균형(CI), 속성 불균형(AI), 가짜 상관관계(SC)**를 체계적으로 분석하고, 이를 해결하기 위한 클라이언트 선택 알고리즘 **$\text{FEDDIVERSE}$**를 제안한다. 각 클라이언트의 데이터 이질성 특성을 3차원 벡터(DHT)로 추정하고, 서로 **보완적이고 직교하는 특성을 가진 클라이언트들을 조합**하여 학습에 참여시킴으로써, 전역 모델의 일반화 성능과 Worst-group 정확도를 획기적으로 향상시켰다. 이 방법은 매우 낮은 통신 비용으로 구현 가능하며, 기존의 다양한 FL 최적화 기법과 결합하여 시너지 효과를 낼 수 있어 실제 이질적인 데이터 환경의 FL 시스템에 적용될 가능성이 매우 높다.