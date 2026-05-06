# Semantic Borrowing for Generalized Zero-Shot Learning

Xiaowei Chen(2021)

## 🧩 Problem to Solve

본 논문은 Generalized Zero-Shot Learning (GZSL)에서 발생하는 클래스 편향성(partiality) 문제를 해결하고자 한다. GZSL의 핵심 난제는 테스트 단계에서 학습 데이터에 포함되지 않은 unseen classes와 이미 학습된 seen classes가 동시에 등장할 때, 분류기가 seen classes에 과도하게 편향되어 unseen classes를 제대로 분류하지 못하는 점이다. 이는 주로 seen과 unseen 클래스 간의 분포 차이로 인한 domain shift 문제에서 기인한다.

특히 본 연구는 Class-Inductive Instance-Inductive (CIII) 설정에 집중한다. CIII 설정은 학습 과정에서 테스트 데이터의 특징(feature)뿐만 아니라 테스트 대상인 unseen classes의 의미론적 정보(semantics)조차 사용할 수 없는 매우 엄격하고 현실적인 제약 조건이다. 기존의 instance-borrowing이나 synthesizing 방식들은 테스트 단계의 semantics를 활용하여 이 문제를 해결하려 했으나, 이는 CIII 설정에서는 적용이 불가능하며, 특히 합성 방식의 경우 생성 이후 분류기를 다시 학습시켜야 하는 번거로움이 있다. 따라서 본 논문의 목표는 CIII 설정 하에서 testing semantics 없이도 seen 클래스에 대한 편향성을 줄이고 GZSL 성능을 향상시키는 새로운 정규화(regularization) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Semantic Borrowing (SB)**이라는 비전이적(non-transductive) 정규화 기법을 도입하는 것이다. SB의 중심 직관은 학습 세트 내에서 특정 클래스의 semantic vector와 가장 유사한 다른 semantic vector를 '빌려와(borrow)', 해당 feature와 빌려온 semantic 간의 compatibility(적합성) 또한 높이도록 학습시키는 것이다.

이를 통해 분류기는 unseen classes의 실제 semantics를 알지 못하더라도, seen 클래스들 간의 semantic 관계를 더 정교하게 모델링함으로써 결과적으로 unseen 클래스의 semantic 영역까지 더 정확하게 추론할 수 있는 능력을 갖게 된다. 이는 결과적으로 seen 클래스로의 편향성을 완화하여 GZSL의 전반적인 성능을 높이는 효과를 가져온다. 또한, SB는 특정 모델에 종속되지 않는 정규화 항이므로 선형 모델뿐만 아니라 인공신경망과 같은 비선형 모델에도 유연하게 적용 가능하다는 강점을 가진다.

## 📎 Related Works

논문에서는 GZSL 방법론을 correspondence, relationship, combination, projection, instance-borrowing, synthesizing의 여섯 가지 그룹으로 분류한다.

기존의 instance-borrowing 및 synthesizing 방법들은 domain shift 문제를 해결하기 위해 테스트 세트의 semantics를 활용하여 가상 데이터를 생성하거나 활용한다. 하지만 이러한 방식들은 CIII 설정(테스트 데이터 및 semantics 접근 불가)에서는 원천적으로 사용할 수 없다는 한계가 있다. 반면, 본 논문에서 제안하는 SB는 오직 학습 데이터 내의 seen semantics만을 활용하므로 CIII 설정을 완전히 준수하면서도 유사한 효과를 거둘 수 있다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 시스템 구조 및 Compatibility Metric Learning

본 방법론은 feature $\mathbf{f}$와 semantic vector $\mathbf{s}$ 사이의 적합성을 측정하는 compatibility function $c(\mathbf{f}, \mathbf{s}; \theta)$를 학습하는 Compatibility Metric Learning을 기반으로 한다. 테스트 단계에서는 특정 feature $\mathbf{f}$에 대해 가장 높은 compatibility 값을 가지는 semantic $\mathbf{s}$를 선택하여 클래스를 결정한다.

$$ \mathcal{M}(\mathbf{f}) = \arg\max_{\mathbf{s} \in \mathcal{S}} c(\mathbf{f}, \mathbf{s}) $$

### 2. 기본 학습 목표 ($\mathcal{L}_0$)

SB를 적용하기 전, 모델의 복잡도에 따라 서로 다른 기본 손실 함수를 정의한다.

**선형 모델 (Linear Model):**
Symmetric structured joint embedding을 사용하여 feature와 semantic 양방향의 오분류 손실을 합산한다.
$$ \mathcal{L}_0(\mathbf{f}_i, \mathbf{s}_i; \theta) = \mathcal{L}_0^f(\mathbf{f}_i, \mathbf{s}_i; \theta) + \mathcal{L}_0^s(\mathbf{f}_i, \mathbf{s}_i; \theta) $$
여기서 각 손실 함수는 다음과 같이 정의된다.
$$ \mathcal{L}_0^f(\mathbf{f}_i, \mathbf{s}_i; \theta) = \frac{\sum_{\mathbf{s} \in \mathcal{S}_{tr} \setminus \{\mathbf{s}_i\}} \max\{0, 1 + c(\mathbf{f}_i, \mathbf{s}; \theta) - c(\mathbf{f}_i, \mathbf{s}_i; \theta)\}}{|\mathcal{S}_{tr}| - 1} $$
$$ \mathcal{L}_0^s(\mathbf{f}_i, \mathbf{s}_i; \theta) = \frac{\sum_{\mathbf{f} \in \mathcal{F}_{tr} \setminus \{\mathbf{f}_i\}} \max\{0, 1 + c(\mathbf{f}, \mathbf{s}_i; \theta) - c(\mathbf{f}_i, \mathbf{s}_i; \theta)\}}{|\mathcal{F}_{tr}| - 1} $$

**비선형 모델 (Nonlinear Model):**
강력한 피팅 능력을 고려하여 MSE(Mean Square Error) 손실을 사용한다.
$$ \mathcal{L}_0(\mathbf{f}_i, \mathbf{s}_i; \theta) = \frac{\sum_{\mathbf{s} \in \mathcal{S}_{tr} \setminus \{\mathbf{s}_i\}} c^2(\mathbf{f}_i, \mathbf{s}; \theta)}{|\mathcal{S}_{tr}| - 1} + [c(\mathbf{f}_i, \mathbf{s}_i; \theta) - 1]^2 $$

### 3. Semantic Borrowing 정규화 ($\mathcal{L}_{SB}$)

SB는 학습 세트 내에서 $\mathbf{s}_i$와 가장 유사한 semantic $\mathbf{s}_j$를 찾아 $\mathbf{f}_i$와 $\mathbf{s}_j$ 간의 compatibility를 높이도록 유도한다.

**선형 모델의 SB:**
$$ \mathcal{L}_{SB}(\mathbf{f}_i, \mathbf{s}_i, \mathbf{s}_j; \theta) = \frac{\sum_{\mathbf{s} \in \mathcal{S}_{tr} \setminus \{\mathbf{s}_j\}} \max\{0, 1 + c(\mathbf{f}_i, \mathbf{s}; \theta) - c(\mathbf{f}_i, \mathbf{s}_j; \theta)\}}{|\mathcal{S}_{tr}| - 1} $$

**비선형 모델의 SB:**
$$ \mathcal{L}_{SB}(\mathbf{f}_i, \mathbf{s}_i, \mathbf{s}_j; \theta) = \frac{\sum_{\mathbf{s} \in \mathcal{S}_{tr} \setminus \{\mathbf{s}_j\}} c^2(\mathbf{f}_i, \mathbf{s}; \theta)}{|\mathcal{S}_{tr}| - 1} + [c(\mathbf{f}_i, \mathbf{s}_j; \theta) - 1]^2 $$

### 4. 전체 손실 함수 및 학습 절차

최종 손실 함수는 기본 손실, SB 정규화 항, 그리고 weight decay 항의 합으로 구성된다.
$$ \mathcal{L}_{tot}(\theta) = \sum_{(\mathbf{f}_i, \mathbf{s}_i) \in \mathcal{D}_{tr}} \mathcal{L}_0(\mathbf{f}_i, \mathbf{s}_i; \theta) + \alpha \sum_{(\mathbf{f}_i, \mathbf{s}_i) \in \mathcal{D}_{tr}} \mathcal{L}_{SB}(\mathbf{f}_i, \mathbf{s}_i, \mathcal{C}(\mathbf{s}_i); \theta) + \beta\|\theta\|^2 $$
여기서 $\alpha$는 정규화의 강도를 조절하는 하이퍼파라미터이며, $\mathcal{C}(\mathbf{s}_i)$는 $\mathbf{s}_i$와 가장 유사한 semantic $\mathbf{s}_j$를 찾는 함수이다.

**Semantic Similarity 측정:**
유사한 semantic을 찾기 위해 Negative Mean Absolute Error (-MAE)를 사용한다.
$$ \mathcal{C}(\mathbf{s}_i) = \arg\min_{\mathbf{s} \in \mathcal{S}_{tr}} \|\mathbf{s} - \mathbf{s}_i\|_1 $$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Fine-grained (CUB, SUN), Coarse-grained (AWA1, AWA2, aPY).
- **특징 추출:** ImageNet-1K로 사전 학습된 ResNet-101의 2048차원 top pooling units 사용.
- **평가 지표:** Unseen class 정확도($u$), Seen class 정확도($s$), 그리고 이들의 조화 평균인 $h$를 사용한다.
- **비교 모델:** 선형 모델(Bilinear mapping), 비선형 모델(MLP combination).

### 2. 주요 결과

- **SOTA 비교:** SB를 적용한 모델들은 대부분의 데이터셋에서 기존의 inductive GZSL 방법들보다 높은 $h$와 $u$ 값을 기록하였다. 특히, CIII 설정을 준수하는 모델들 사이에서 최상위 성능을 보였다.
- **효과성 검증 (Ablation Study):** SB를 적용하지 않았을 때보다 적용했을 때 $u$와 $h$가 일관되게 향상되었으며, 일부 사례에서는 $s$ 또한 개선되었다. 이는 SB가 seen과 unseen 클래스 간의 관계를 더 정확하게 모델링함을 입증한다.
- **하이퍼파라미터 $\alpha$의 영향:** 실험 결과 $\alpha \ge 1$인 경우 오히려 성능이 하락하였다. 이는 borrowed semantic의 compatibility가 실제 해당 클래스의 compatibility보다 커질 경우 모델링이 왜곡되기 때문이다. 최적의 값은 데이터셋과 지표에 따라 $\alpha=0.01$(best $h, s$)에서 $\alpha=0.1$(best $h, u$) 사이에 존재한다.

## 🧠 Insights & Discussion

본 논문은 GZSL의 고질적인 문제인 seen class 편향성을 해결하기 위해, 외부 데이터나 unseen semantics 없이 오직 내부 semantic 관계를 이용한 정규화만으로 성능을 높일 수 있음을 보여주었다. 특히 CIII라는 매우 제한적인 설정에서도 작동한다는 점이 실용적인 강점이다.

비판적으로 해석하자면, SB의 성능 향상이 $\alpha$라는 하이퍼파라미터에 매우 민감하게 반응한다는 점이 한계로 지적될 수 있다. $\alpha$ 값이 너무 크면 오히려 성능이 급격히 저하되는데, 이는 정규화 항이 주 목적 함수를 압도하여 발생하는 현상이다. 따라서 최적의 $\alpha$를 찾기 위한 추가적인 탐색 과정이 필요하다. 또한, 본 논문은 compatibility metric learning 기반의 모델에 집중하고 있으나, 이를 generative 모델과 결합했을 때 어떤 시너지가 날지에 대한 연구는 향후 과제로 남아있다.

## 📌 TL;DR

본 논문은 CIII 설정의 Generalized Zero-Shot Learning에서 seen 클래스로의 편향성을 줄이기 위해, 학습 세트 내의 유사 semantic을 활용하는 **Semantic Borrowing (SB)** 정규화 기법을 제안하였다. SB는 별도의 추가 정보 없이도 seen과 unseen 클래스 간의 의미론적 관계를 정교하게 학습하게 하며, 선형 및 비선형 모델 모두에 적용 가능하여 기존 inductive GZSL의 SOTA 성능을 경신하였다. 이 연구는 현실적인 제약 조건 하에서 GZSL의 강건성을 높이는 효율적인 방법론을 제시했다는 점에서 의의가 있다.
