# MULTI-SOURCE-FREE DOMAIN ADAPTATION VIA UNCERTAINTY-AWARE ADAPTIVE DISTILLATION

Yaxuan Song, Jianan Fan, Dongnan Liu, Weidong Cai (2024/2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석 분야에서 발생하는 Multi-Source-Free Domain Adaptation (MSFDA) 문제를 해결하고자 한다. 의료 데이터는 개인정보 보호 및 규제로 인해 학습 데이터(Source data)를 직접 공유하는 것이 엄격히 제한된다. 따라서 소스 데이터 없이 사전 학습된 모델들만 제공되는 Source-Free Domain Adaptation (SFDA) 설정이 필수적이다.

특히 의료 데이터는 여러 의료 기관(Multi-centre)에서 서로 다른 장비와 프로토콜을 통해 수집되므로 기관 간의 도메인 차이(Domain discrepancy)가 매우 크다. 기존의 SFDA 또는 MSFDA 방법론들은 이러한 의료 데이터의 특수성으로 인해 타겟 도메인으로의 일반화 능력이 떨어지며, 부적절한 소스 모델을 사용할 경우 Negative transfer가 발생하여 오히려 성능이 저하되는 문제가 있다. 본 연구의 목표는 불확실성 인지 기반의 적응형 증류(Uncertainty-aware Adaptive Distillation, UAD)를 통해 신뢰도 높은 지식 전이를 달성하고 타겟 도메인 모델의 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델 수준(Model-level)과 인스턴스 수준(Instance-level)이라는 두 가지 상호 보완적인 관점에서 Uncertainty-aware Adaptive Distillation (UAD)을 수행하여 지식 증류의 신뢰성을 높이는 것이다.

1. **두 단계의 UAD 프레임워크**: 타겟 도메인의 데이터 분포와 가장 유사한 소스 모델을 선택하여 초기화를 수행하는 '모델 수준 UAD'와, 개별 데이터마다 최적의 소스 모델을 선택해 고품질의 pseudo-label을 생성하는 '인스턴스 수준 UAD'를 제안한다.
2. **Temperature Scaling (TS)을 통한 보정**: 모델의 과잉 신뢰(Over-confidence) 또는 과소 신뢰(Under-confidence) 문제를 해결하기 위해 Temperature Scaling을 도입하여, 신뢰도 측정 전 Logit을 보정함으로써 더 정확한 불확실성 추정을 가능하게 한다.
3. **의료 영상 데이터셋 검증**: Diabetic Retinopathy (DR) 및 Skin Cancer (HAM10000)와 같은 다기관 의료 데이터셋을 통해 제안 방법론의 실용성과 성능 우위를 입증하였다.

## 📎 Related Works

기존의 Unsupervised Domain Adaptation (UDA)은 소스 데이터와 타겟 데이터를 동시에 활용하여 도메인 간 간극을 줄이는 방식을 취했다. 그러나 데이터 프라이버시 문제로 인해 소스 데이터 없이 모델만 사용하는 SFDA가 등장하였다. 최근에는 여러 개의 소스 모델을 활용하는 MSFDA 연구(예: DECISION, CAiDA)가 진행되었으나, 이들은 주로 self-supervised clustering 기반의 pseudo-labelling에 의존하는 경향이 있다.

본 논문은 기존 MSFDA 접근 방식들이 의료 영상의 큰 도메인 편차를 충분히 극복하지 못하며, 부적절한 소스 모델의 개입으로 인해 타겟 모델이 노이즈 섞인 라벨에 과적합(Overfitting)되는 한계가 있음을 지적한다. 제안된 UAD는 단순히 모든 모델을 가중 합산하는 대신, 불확실성 측정 기반으로 최적의 모델을 선택적으로 활용함으로써 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 문제 정의 및 설정

$N$개의 소스 도메인에서 학습된 모델 족(Model zoo) $\{\theta_S^j\}_{j=1}^N$과 라벨이 없는 타겟 데이터셋 $D_T = \{x_i^T\}_{i=1}^{n_T}$가 주어진 상태에서, 타겟 분류 모델 $\theta_T$를 학습시키는 것을 목표로 한다.

### 2. 신뢰도 측정 지표: Margin

모델의 예측 신뢰도를 측정하기 위해 Margin $M$을 사용한다. Margin은 Softmax 결과값 중 가장 확률이 높은 클래스와 두 번째로 높은 클래스의 차이로 정의된다.

$$M = \text{Top}_1(\delta(\theta(x))) - \text{Top}_2(\delta(\theta(x)))$$

여기서 $\delta(\cdot)$는 Softmax 함수를 의미한다. Margin 값이 클수록 해당 모델이 해당 인스턴스에 대해 더 확신을 가지고 예측한 것으로 간주한다.

### 3. Model-level UAD (초기화 단계)

부적절한 소스 모델로 인한 Negative transfer를 방지하기 위해, 타겟 도메인 전체 데이터에 대해 가장 높은 평균 신뢰도를 보이는 모델을 선택하여 초기 모델 $\theta_T$로 설정한다.

각 소스 모델 $\theta_S^j$의 평균 신뢰도 $M_j$는 다음과 같다.
$$M_j = \frac{1}{n_T} \sum_{i=1}^{n_T} M_i$$

최종적으로 가장 큰 $M_j$를 갖는 모델 $\theta_S^*$를 선택한다.
$$\epsilon = \arg \max([M_j]_{j=1}^N)$$
$$\theta_T \leftarrow \theta_S^*$$

### 4. Instance-level UAD (학습 단계)

타겟 데이터의 각 인스턴스 $x_i^T$에 대해, 모델 족 내에서 가장 높은 Margin을 가진 모델을 선택하여 pseudo-label $\hat{y}_i^T$를 생성한다.

$$\epsilon^i = \arg \max([M_i]_{j=1}^N)$$
$$\hat{y}_i^T = \theta_{\epsilon^i}(x_i^T)$$

생성된 pseudo-label을 사용하여 초기화된 타겟 모델 $\theta_T$를 다음과 같은 Cross-Entropy 손실 함수 $\mathcal{L}_{tar}$로 미세 조정(Fine-tuning)한다.

$$\mathcal{L}_{tar} = -\mathbb{E}_{(x_T, \hat{y}_T) \in X_T \times \hat{Y}_T} \sum_{k=1}^K \mathbb{1}[k = \hat{y}_T] \log \delta_k(\theta_T(x_T))$$

### 5. Temperature Scaling (TS)

모델의 예측 확률과 실제 정확도 사이의 불일치(Mismatch)를 해결하기 위해 Logit을 보정한다. 각 소스 모델에 대해 온도 파라미터 $T_j$를 도입하여 다음과 같이 보정된 Logit $z_j$를 계산한다.

$$z_j = \theta_S^j(x_i^T) / T_j$$

$T_j$는 Expected Calibration Error (ECE)를 최소화하도록 학습된다.
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n_T} |\text{acc}(B_m) - \text{conf}(B_m)|$$

## 📊 Results

### 실험 설정

- **데이터셋**:
  - Diabetic Retinopathy (DR): APTOS 2019, DDR, IDRiD (3개 도메인)
  - Skin Cancer (HAM10000): 신체 부위(Back, Face, Lower extremity, Upper extremity)별 4개 도메인
- **백본**: DenseNet-121
- **비교 대상**: AaD (Multi-source 확장 버전), DECISION, CAiDA
- **지표**: Adaptation Accuracy (%)

### 주요 결과

실험 결과, 제안된 UAD 방법론이 모든 데이터셋에서 기존 baseline들을 유의미하게 상회하는 성능을 보였다.

- **정량적 결과**: DR 데이터셋과 HAM10000 데이터셋 모두에서 평균 정확도가 가장 높게 나타났다. 특히 타겟 도메인의 특성이 까다로운 경우(예: DR의 I 도메인, HAM10000의 F 도메인)에 더 큰 성능 향상을 보였다.
- **Ablation Study 결과**:
  - **M-UAD만 적용**: Baseline 대비 약 5%의 평균 성능 향상이 있었으며, 이는 적절한 모델 초기화의 중요성을 보여준다.
  - **I-UAD만 적용**: M-UAD보다 더 높은 정확도를 보였으며, 고품질 pseudo-label의 효과를 입증한다.
  - **M-UAD + I-UAD**: 두 방법을 동시에 적용했을 때 시너지 효과가 발생하여 성능이 더욱 향상되었다.
  - **TS 적용**: Temperature Scaling을 추가했을 때, 특히 정확도가 낮았던 도메인에서 뚜렷한 성능 개선이 확인되어 신뢰도 보정의 필요성이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 MSFDA 설정에서 단순히 여러 모델의 지식을 통합하는 것이 아니라, **'누가 가장 믿을만한가'**를 모델 수준과 인스턴스 수준에서 각각 판단하는 전략이 유효함을 보여주었다. 특히 의료 영상과 같이 도메인 간 편차가 극심한 환경에서는 모든 소스 모델이 도움이 되는 것이 아니라, 일부 모델이 오히려 독이 되는 Negative transfer를 일으킬 수 있다는 점을 잘 파악하였다.

**강점**으로는 복잡한 아키텍처 변경 없이 Margin 기반의 선택 전략과 Temperature Scaling이라는 비교적 단순한 기법만으로 높은 성능 향상을 이끌어냈다는 점이다.

**한계 및 논의사항**으로는, Temperature Scaling을 위해 타겟 도메인 데이터 $D_T$를 사용하여 $T_j$를 최적화하는데, 이 과정에서 타겟 데이터의 분포를 얼마나 효율적으로 반영했는지에 대한 추가 분석이 필요해 보인다. 또한, 분류 문제 외에 세그멘테이션이나 검출과 같은 다른 의료 영상 작업으로의 확장 가능성에 대해서는 명시적으로 다루지 않았다.

## 📌 TL;DR

본 논문은 의료 영상 분석을 위한 Multi-Source-Free Domain Adaptation (MSFDA) 프레임워크인 **UAD (Uncertainty-aware Adaptive Distillation)**를 제안한다. 이 방법은 $\text{Top}_1$과 $\text{Top}_2$ 확률의 차이인 Margin을 이용해 **(1) 타겟 도메인에 가장 적합한 모델로 초기화(Model-level)**하고, **(2) 각 데이터별로 최적의 모델을 통해 pseudo-label을 생성(Instance-level)**하여 학습한다. 여기에 **Temperature Scaling**을 통한 신뢰도 보정을 추가하여 Negative transfer를 억제했다. 결과적으로 다기관 의료 데이터셋에서 기존 MSFDA 방법론들보다 우수한 성능을 기록하며, 의료 데이터 프라이버시를 준수하면서도 도메인 적응을 달성할 수 있는 가능성을 제시하였다.
