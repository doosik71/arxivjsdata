# CAKD: A Correlation-Aware Knowledge Distillation Framework Based on Decoupling Kullback-Leibler Divergence

Zao Zhang, Huaming Chen, Pei Ning, Nan Yang, Dong Yuan (2024)

## 🧩 Problem to Solve

본 논문은 지식 증류(Knowledge Distillation, KD) 과정에서 증류 구성 요소(distillation component)를 하나의 단일 단위로 처리함으로써 발생하는 문제점에 주목한다. 기존의 KD 방법론들은 손실 함수 내의 개별 요소들이 모델의 예측에 미치는 세부적인 영향력을 간과하는 경향이 있으며, 이는 증류 과정의 효율성과 학생 모델(student model)의 최종 성능을 최적화하는 데 한계로 작용한다.

특히, 모든 특징(feature)이나 로짓(logit)이 최종 출력 결정에 동일한 중요도를 가지지 않는다는 점에 착안하여, 어떤 요소가 결정적인 영향을 미치고 어떤 요소가 중복되거나 덜 중요한지를 구분하여 학습시키는 것이 중요하다. 따라서 본 연구의 목표는 KL Divergence를 수학적으로 분해하여 각 구성 요소의 물리적 의미를 부여하고, 영향력이 큰 요소에 우선순위를 두어 지식 전송을 최적화하는 Correlation-Aware Knowledge Distillation (CAKD) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 KL Divergence를 세 가지 고유한 요소인 Binary Classification Divergence (BCD), Strong Correlation Divergence (SCD), 그리고 Weak Correlation Divergence (WCD)로 디커플링(decoupling)하는 것이다. 

중심적인 직관은 모델의 예측에 강하게 연관된 특징(Strong Correlation)과 약하게 연관된 특징(Weak Correlation)이 학생 모델에 전달해야 하는 지식의 성격이 다르다는 점이다. CAKD는 이러한 상관관계 기반의 분류를 통해 각 요소의 가중치를 조절함으로써, 학생 모델이 고품질의 정보를 선택적으로 학습하게 하여 예측 성능을 향상시킨다.

## 📎 Related Works

기존의 지식 증류 연구는 크게 두 가지 범주로 나뉜다. 첫째는 로짓 기반(logit-based) KD로, 교사 모델의 출력 로짓(soft labels)을 모방하여 클래스 간의 관계 정보를 학습하는 방식이다. 둘째는 특징 기반(feature-based) KD로, 중간 계층의 특징 표현이나 활성화 맵을 일치시켜 교사 모델의 내부 추론 과정을 학습하는 방식이다.

기존 접근 방식들은 주로 여러 증류 구성 요소를 어떻게 조합하고 균형을 맞출 것인가에 집중했지만, 개별 구성 요소 내부의 세부 요소들이 예측에 미치는 영향력을 분석하는 데는 미흡했다. CAKD는 이러한 한계를 극복하기 위해 로짓과 특징 모두에 적용 가능한 디커플링 메커니즘을 도입하여, 단순한 일치(matching)를 넘어 상관관계에 따른 차등적 학습을 가능하게 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
CAKD 프레임워크는 교사 네트워크와 학생 네트워크의 특징($F_T, F_S$) 및 로짓($Z_T, Z_S$)을 입력으로 받는다. 제안된 시스템은 먼저 특징들을 예측과의 상관관계에 따라 강한 상관관계 클러스터($S$)와 약한 상관관계 클러스터($W$)로 분류하며, 이 분류의 정확도를 나타내는 이진 분류 항목을 포함한다. 최종적으로 CAKD 손실 함수는 특징 수준과 로짓 수준에서 디커플링된 KL Divergence들의 가중치 합으로 구성된다.

### KL Divergence 디커플링 메커니즘
기본적인 KL Divergence는 다음과 같이 정의된다:
$$KD = KL(p^T \parallel p^S) = \sum_{i=1}^{C} p^T_i \log \left( \frac{p^T_i}{p^S_i} \right)$$
여기서 $p$는 softmax 함수를 통해 계산된 확률 분포이다. 

본 논문에서는 특징들을 강한 상관관계 집합 $S$와 약한 상관관계 집합 $W$로 나누고, 이에 따라 확률 $p$를 다음과 같이 재정의한다. 강한 상관관계 클러스터의 합산 확률을 $p_s$, 약한 상관관계 클러스터의 합산 확률을 $p_w$라고 할 때, 각 요소 $p_i$는 다음과 같이 표현될 수 있다:
$$p_i = \begin{cases} p_s \cdot \hat{p}_i & \text{if } i \in S \\ p_w \cdot \hat{p}_i & \text{if } i \notin S \end{cases}$$
여기서 $\hat{p}_i$는 각 클러스터 내부에서의 상대적인 확률 분포를 의미한다.

이러한 정의를 바탕으로 KL Divergence를 수학적으로 분해하면 다음과 같은 최종 식을 얻는다:
$$KD = KL(b^T \parallel b^S) + p^T_s KL(\hat{p}^T_s \parallel \hat{p}^S_s) + p^T_w KL(\hat{p}^T_w \parallel \hat{p}^S_w)$$
이를 각각 다음과 같이 명명한다:
- **BCD (Binary Classification Divergence)**: $KL(b^T \parallel b^S)$, 강한 상관관계와 약한 상관관계 집합 사이의 이진 분류 정확도를 측정한다.
- **SCD (Strong Correlation Divergence)**: $KL(\hat{p}^T_s \parallel \hat{p}^S_s)$, 강한 상관관계 집합 내부에서의 특징 분포 차이를 측정한다.
- **WCD (Weak Correlation Divergence)**: $KL(\hat{p}^T_w \parallel \hat{p}^S_w)$, 약한 상관관계 집합 내부에서의 특징 분포 차이를 측정한다.

결과적으로 전체 손실 함수는 다음과 같이 요약된다:
$$KD = BCD + p^T_s SCD + p^T_w WCD$$

### 학습 절차 및 특수 케이스
- **단일 정답 레이블의 로짓 증류**: 정답 레이블이 하나인 경우, 강한 상관관계 클러스터 $S$는 단 하나의 요소만 포함하게 된다. 이 경우 SCD는 BCD에 포함되어 사라지며, 최종 식은 $KD = BCD + p^T_w WCD$가 된다.
- **가중치 조절**: 표준 KL Divergence에서는 $p^T_s$와 $p^T_w$에 의해 SCD와 WCD가 억제되는 경향이 있다. CAKD는 하이퍼파라미터 $\alpha$와 $\beta$를 도입하여 SCD와 WCD의 영향을 강화함으로써, 더 어렵고 정교한 지식을 전달하도록 유도한다.

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR-100, Tiny-ImageNet, ImageNet을 사용하여 모델의 범용성을 검증하였다.
- **백본 모델**: ResNet, WideResNet(WRN), ShuffleNet, MobileNet 등 다양한 구조를 사용하였다.
- **비교 대상 (Baselines)**: KD, CRD, OFD, CTKD, ReviewKD, DKD, NKD 등의 최신 KD 방법론과 비교하였다.
- **지표**: Top-1 Accuracy를 측정 지표로 사용하였다.

### 주요 결과
- **정량적 성능**: CIFAR-100, Tiny-ImageNet, ImageNet 모든 데이터셋에서 CAKD가 기존 베이스라인 모델들보다 일관되게 높은 정확도를 달성하였다. 특히 ImageNet과 같은 대규모 데이터셋에서도 우수한 성능을 보여 방법론의 확장성을 입증하였다.
- **상관관계 분석**: CIFAR-100과 같이 교사 모델의 정확도가 높은 경우 $\alpha$(SCD 가중치)의 영향이 더 컸으며, Tiny-ImageNet과 같이 교사 모델의 신뢰도가 상대적으로 낮은 경우에는 $\beta$(WCD 가중치)를 적절히 유지하는 것이 성능 향상에 중요했다.
- **구성 요소의 조합**: 로짓 증류와 특징 증류를 동시에 수행했을 때 가장 높은 성능이 나타났으며, 증류하는 레이어의 수를 늘릴수록 성능이 향상되는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 KD의 손실 함수를 세분화하여 분석함으로써, 단순히 교사 모델을 모방하는 것이 아니라 '어떤 정보가 더 중요한가'에 따라 학습 강도를 조절하는 것이 효과적임을 보여주었다. 특히 교사 모델의 신뢰도(Confidence)가 특징 마스크(Strong vs Weak) 생성의 정확도에 영향을 미치며, 이에 따라 SCD와 WCD의 중요도가 달라진다는 통찰을 제시하였다.

**강점**으로는 수학적인 기반 위에 KL Divergence를 손실 없이 분해하여 물리적 의미를 부여했다는 점과, 이를 통해 하이퍼파라미터 튜닝의 근거를 마련했다는 점을 들 수 있다.

**한계 및 논의사항**으로는, 특징을 $S$와 $W$로 나누는 클러스터링 과정이 교사 모델의 신뢰도에 지나치게 의존한다는 점이 언급된다. 교사 모델의 예측이 부정확할 경우 잘못된 마스크가 생성되어 오히려 학습을 방해할 가능성이 있으며, 이는 Tiny-ImageNet 결과에서 개선 폭이 CIFAR-100보다 작았던 원인으로 분석된다. 또한, 현재는 단일 레이블 분류 작업에 집중되어 있어, 다중 클래스 분류(multi-class classification) 시나리오에서의 효과는 향후 과제로 남아 있다.

## 📌 TL;DR

CAKD는 KL Divergence를 BCD, SCD, WCD라는 세 가지 상관관계 기반 요소로 분해하여, 예측에 결정적인 영향을 미치는 지식을 선택적으로 강화해 전달하는 지식 증류 프레임워크이다. 이 연구는 증류 구성 요소의 세부 분석이 성능 향상에 필수적임을 입증하였으며, 다양한 데이터셋과 모델 구조에서 기존 SOTA 방법론들을 능가하는 성능을 보였다. 이는 향후 모델 압축 및 전이 학습 연구에서 지식의 '중요도'를 정교하게 제어하는 새로운 방향성을 제시한다.