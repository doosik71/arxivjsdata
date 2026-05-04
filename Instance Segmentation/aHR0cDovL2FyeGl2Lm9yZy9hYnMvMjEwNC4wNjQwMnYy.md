# DropLoss for Long-Tail Instance Segmentation

Ting-I Hsieh, Esther Robb, Hwann-Tzong Chen, Jia-Bin Huang (2021)

## 🧩 Problem to Solve

본 논문은 인스턴스 분할(Instance Segmentation) 작업에서 발생하는 클래스 불균형 문제, 특히 롱테일(Long-tailed) 분포 문제를 해결하고자 한다. 실제 환경의 데이터셋은 소수의 빈번한 클래스(Frequent categories)가 대부분의 데이터를 차지하고, 다수의 희귀 클래스(Rare categories)는 매우 적은 수의 샘플만을 가지는 특성을 보인다.

이러한 분포에서 모델을 학습시키면 빈번한 클래스에 편향된 예측을 하는 경향이 있으며, 희귀 클래스의 경우 데이터 부족으로 인해 과적합(Overfitting) 위험이 크다. 특히 저자들은 희귀 클래스의 예측 확률이 '정확한 배경 예측(Correct background predictions)'에 의해 심하게 억제된다는 점에 주목한다. 배경 영역으로 분류되어야 할 영역(Background anchors)에 대해 모든 전경 클래스의 확률을 낮추는 손실 함수가 적용되는데, 희귀 클래스는 이를 상쇄할 '긍정적인 그래디언트(Encouraging gradients)'가 매우 적기 때문에 결과적으로 빈번한 클래스보다 더 심하게 억제되어 예측 성능이 저하된다. 따라서 본 논문의 목표는 희귀 및 보통(Common) 클래스가 배경 예측 손실에 의해 과도하게 억제되지 않도록 하는 적응형 손실 함수를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **배경 예측에서 발생하는 '저해하는 그래디언트(Discouraging gradients)'를 적응적으로 제거**하여 클래스 간의 불균형을 해소하는 것이다.

저자들은 배경 영역에 대한 손실 계산 시, 해당 배치의 통계 정보를 활용하여 희귀 및 보통 클래스의 손실을 확률적으로 제거하는 **DropLoss**를 제안한다. 이는 빈번한 클래스와 희귀 클래스 사이의 성능 트레이드-오프(Trade-off) 없이 전체적인 성능을 향상시키는 단순하면서도 효과적인 방법이다.

## 📎 Related Works

롱테일 분포를 해결하기 위한 기존 접근 방식은 크게 세 가지로 분류된다:
1. **리샘플링(Resampling):** 희귀 클래스를 오버샘플링하거나 빈번한 클래스를 언더샘플링하여 데이터 분포를 맞추는 방식이다. 하지만 인스턴스 분할에서는 이미지당 클래스 구성이 달라 적용이 어렵다.
2. **재가중치 및 비용 민감 학습(Reweighting and Cost-sensitive Learning):** 클래스 빈도의 역수를 사용하여 손실 가중치를 조정하는 방식이다.
3. **특징 조작(Feature Manipulation):** 정규화나 메트릭 학습을 통해 특징 공간에서 클래스 간 거리를 조정하는 방식이다.

특히 기존의 최신 기법인 **Equalization Loss (EQL)**는 잘못된 전경 클래스 예측(Incorrect foreground classification)에서 발생하는 저해하는 그래디언트를 제거하여 희귀 클래스를 보호했다. 그러나 본 논문은 EQL이 간과한 '배경 클래스 예측'에서 오는 그래디언트가 실제로는 훨씬 더 지배적인 영향을 미친다는 점을 분석하여 차별점을 둔다.

## 🛠️ Methodology

### 1. Baseline: Equalization Loss (EQL)
먼저 기본이 되는 $\text{L}_{\text{EQL}}$은 다음과 같이 정의된다:
$$\text{L}_{\text{EQL}} = -\sum_{j=1}^{C} w_j \log(\hat{p}_j)$$
여기서 $\hat{p}_j$는 정답 라벨 $y_j$에 따른 예측 확률이며, 가중치 $w_j$는 다음과 같다:
$$w_j = 1 - \text{E}(r) \text{T}_{\lambda}(f_j)(1 - y_j)$$
$\text{E}(r)$은 전경 영역일 때 1, 배경일 때 0을 반환하는 지시 함수이며, $\text{T}_{\lambda}(f_j)$는 클래스 $j$의 빈도 $f_j$가 임계값 $\lambda$보다 낮으면 1을 반환하는 함수이다. 즉, 전경 영역($\text{E}(r)=1$)에서 희귀 클래스($\text{T}_{\lambda}=1$)에 대한 잘못된 예측 시 가중치가 0이 되어 패널티를 제거한다.

### 2. Background Equalization Loss ($\text{L}_{\text{BEQL}}$)
저자들은 배경 영역에서의 억제 효과를 줄이기 위해 $\text{L}_{\text{BEQL}}$을 제안했다. 배경 영역($\text{E}(r)=0$)에 대해 예측 확신도(Confidence)가 낮을 때 가중치를 더 낮게 부여하는 방식이다:
$$w_j = \begin{cases} 1 - \text{T}_{\lambda}(f_j)(1 - y_j), & \text{if } \text{E}(r) = 1 \\ 1 - \text{T}_{\lambda}(f_j) \cdot \min\{-\log_b(p_j), 1\}, & \text{otherwise} \end{cases}$$
여기서 $\log$의 밑인 $b$는 가중치 민감도를 조절하는 하이퍼파라미터이다. 하지만 이 방식은 $b$ 값에 따라 빈번한 클래스와 희귀 클래스의 성능이 반비례하는 트레이드-오프 문제가 발생한다.

### 3. Proposed Method: DropLoss
트레이드-오프 문제를 해결하기 위해, 저자들은 배경 영역의 손실 가중치를 베르누이 분포(Bernoulli distribution)에서 샘플링하는 **DropLoss**를 제안한다:
$$\text{L}_{\text{Drop}} = -\sum_{j=1}^{C} w_j \log(\hat{p}_j)$$
가중치 $w_j$는 다음과 같이 결정된다:
$$w_j = \begin{cases} 1 - \text{T}_{\lambda}(f_j)(1 - y_j), & \text{if } \text{E}(r) = 1 \\ w \sim \text{Ber}(\mu_{f_j}), & \text{otherwise} \end{cases}$$
여기서 $\mu_{f_j}$는 현재 학습 배치(Batch) 내의 클래스 발생 비율에 따라 동적으로 결정된다:
$$\mu_{f_j} = \begin{cases} (n_{\text{rare}} + n_{\text{common}}) / n_{\text{all}}, & \text{if } \text{T}_{\lambda}(f_j) = 1 \\ n_{\text{frequent}} / n_{\text{all}}, & \text{otherwise} \end{cases}$$
- $n_{\text{rare}}, n_{\text{common}}, n_{\text{frequent}}$: 현재 배치 내 전경 영역에서 각 클래스의 출현 횟수.
- $n_{\text{all}}$: 전체 전경 영역의 출현 횟수 합.

**동작 원리:** 특정 배치에 희귀 클래스가 적게 포함되어 있다면, 해당 클래스에 대한 배경 손실 가중치 $w_j$가 0이 될 확률이 높아진다. 반대로 해당 클래스가 배치에 많이 포함되어 있다면 손실을 유지할 확률이 높아진다. 이를 통해 모델이 현재 보고 있는 샘플에 적응적으로 집중하게 하며, 불필요한 배경 그래디언트에 의한 억제를 방지한다.

## 📊 Results

### 실험 설정
- **데이터셋:** LVIS v0.5 및 v1.0 (1,230개 클래스, 롱테일 분포).
- **아키텍처:** Mask R-CNN, Cascade R-CNN.
- **백본:** ResNet-50, ResNet-101.
- **평가 지표:** $\text{AP}$ (Average Precision), $\text{AR}$ (Average Recall), 그리고 클래스 그룹별 $\text{AP}_r$ (Rare), $\text{AP}_c$ (Common), $\text{AP}_f$ (Frequent).

### 주요 결과
- **전반적 성능 향상:** 모든 아키텍처와 백본 설정에서 DropLoss가 BCE(기본 손실 함수) 및 EQL보다 높은 $\text{AP}$와 $\text{AR}$을 기록했다.
- **희귀 클래스 개선:** 특히 $\text{AP}_r$과 $\text{AP}_c$에서 괄목할 만한 향상을 보였다. 예를 들어, Mask R-CNN (ResNet-50) 설정에서 $\text{AP}_r$이 EQL 대비 크게 상승하여 전체 $\text{AP}$를 1.7%p 가량 끌어올렸다.
- **리샘플링 기법과의 결합:** Repeat Factor Sampling (RFS)과 결합했을 때 추가적인 성능 향상이 확인되었으며, 이는 데이터 레벨의 리샘플링과 손실 함수 레벨의 재가중치가 상호 보완적임을 시사한다.
- **트레이드-오프 해소:** Pareto Frontier 분석 결과, $\text{L}_{\text{BEQL}}$과 달리 DropLoss는 하이퍼파라미터 튜닝 없이도 빈번한 클래스의 성능 하락을 최소화하면서 희귀 클래스의 성능을 효과적으로 높였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 롱테일 문제의 원인을 '전경 클래스 간의 오분류'가 아닌 '배경 영역에 의한 과도한 억제'에서 찾았다는 분석적 통찰력에 있다. 특히 배경 영역의 손실을 무작정 제거하는 것이 아니라, 배치의 통계 정보를 이용해 확률적으로 제거함으로써 학습의 안정성을 유지했다.

실험 결과에서 배경 그래디언트를 줄였음에도 불구하고 배경과 전경을 구분하는 능력(Background/Foreground classification)은 크게 저하되지 않았는데, 이는 전경-배경 구분 작업이 전경 내 클래스 간 구분 작업보다 훨씬 쉽기 때문으로 해석된다.

한계점으로는 여전히 $\text{AP}_f$ (빈번한 클래스)의 성능이 미세하게 감소하는 경향이 있다는 점이다. 하지만 이는 희귀 클래스의 비약적인 성능 향상으로 상쇄되어 전체적인 $\text{mAP}$는 상승하는 결과를 낳았다.

## 📌 TL;DR

본 논문은 인스턴스 분할의 롱테일 문제에서 **배경 예측 손실이 희귀 클래스를 과도하게 억제**한다는 점을 발견하고, 이를 해결하기 위해 배치 통계 기반의 적응형 손실 제거 기법인 **DropLoss**를 제안하였다. 이 방법은 하이퍼파라미터 튜닝 없이도 희귀/보통 클래스의 성능을 크게 향상시키며, LVIS 데이터셋에서 SOTA 성능을 달성하였다. 이는 향후 다른 롱테일 시각 인식 작업에도 광범위하게 적용될 가능성이 높다.