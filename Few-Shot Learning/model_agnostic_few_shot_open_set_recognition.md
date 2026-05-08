# Model-Agnostic Few-Shot Open-Set Recognition

Malik Boudiaf, Etienne Bennequin, Myriam Tami, Celine Hudelot, Antoine Toubhans, Pablo Piantanida, Ismail Ben Ayed (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Few-Shot Open-Set Recognition (FSOSR)이다. 이는 매우 적은 수의 레이블된 샘플만 주어진 상태에서, 주어진 클래스(Closed-set)에 속하는 인스턴스를 분류함과 동시에, 어떤 알려진 클래스에도 속하지 않는 외부 인스턴스(Open-set)를 탐지하는 과업이다.

현실 세계의 데이터는 대부분 Open-set 특성을 가지므로, 닫힌 집합만을 가정하는 기존의 Few-Shot Classification (FSC) 방식은 외부 인스턴스가 유입되었을 때 이를 강제로 알려진 클래스 중 하나로 잘못 분류하는 치명적인 문제가 있다. 특히 FSC 환경에서는 지원 집합(Support set)의 크기가 매우 작기 때문에, 모델이 클래스 경계를 명확히 정의하기 어려워 Open-set 탐지가 더욱 까다롭다. 따라서 본 논문의 목표는 특정 아키텍처나 훈련 절차에 종속되지 않고, 기존의 사전 학습된 모델에 즉시 적용 가능한 Model-agnostic한 FSOSR 추론 방법을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 추론 단계에서 사용할 수 있는 Model-agnostic한 전이 학습 기반의 방법론인 Open Set Transductive Information Maximization (OSTIM)을 제안한 것이다.

중심적인 설계 아이디어는 다음과 같다. 첫째, 기존의 Transductive 방법들이 쿼리 집합의 모든 샘플을 Closed-set으로 간주하여 외부 인스턴스의 변별력을 떨어뜨리는 문제를 해결하기 위해, 가상의 'Outlier category'를 도입하였다. 둘째, 추가적인 학습 파라미터를 최소화하기 위해 명시적인 프로토타입을 학습시키는 대신, 기존 Closed-set 프로토타입들의 평균과 정반대 방향에 위치하는 Implicit outlier prototype을 정의하여 계산 효율성과 안정성을 높였다. 셋째, 정보 최대화(InfoMax) 원리를 적용하여 쿼리 샘플들이 Inlier category 또는 Outlier group 중 하나로 명확히 분류되도록 유도함으로써, 분류 정확도와 외부 탐지 성능을 동시에 향상시켰다.

## 📎 Related Works

### Few-Shot Classification (FSC)

기존의 FSC 연구들은 주로 Episodic training을 통해 새로운 클래스에 대한 강건한 표현력을 학습시키려 했다. 그러나 최근 연구들은 단순한 Fine-tuning 기반의 Baseline이 복잡한 Episodic 방법들과 경쟁 가능한 수준임을 보여주었으며, 이는 모델 아키텍처에 구애받지 않는 Model-agnostic 방법론의 필요성을 시사한다.

### Transductive FSC

전이 학습(Transduction) 방식은 레이블이 없는 쿼리 집합을 함께 활용하여 분류 성능을 높인다. Laplacian regularization, Clustering, Mutual Information Maximization 등이 사용되었으나, 이들은 기본적으로 모든 쿼리 샘플이 지원 집합의 클래스 내에 존재한다는 Closed-set 가정을 전제로 하므로 Outlier 탐지 성능이 현저히 떨어진다는 한계가 있다.

### Open-Set Recognition (OSR)

OSR은 대규모 데이터셋 환경에서 미지의 클래스를 탐지하는 것을 목표로 한다. OpenMax나 PROSER 같은 방법들이 제안되었으나, 이들은 특정 클래스 집합에 대해 심층 신경망을 직접 훈련시켜야 하므로 샘플 수가 극도로 적은 Few-shot 환경에 직접 적용하기 어렵다.

### Few-Shot Open-Set Recognition (FSOSR)

최근 FSOSR을 해결하기 위해 Meta-learning이나 Transformation consistency를 이용한 방법들이 제안되었다. 하지만 이러한 방법들은 여전히 특수한 Episodic training 전략을 요구하며, 새로운 아키텍처에 적용할 때마다 번거로운 하이퍼파라미터 최적화 과정이 필요하다는 단점이 있다.

## 🛠️ Methodology

### 전체 파이프라인

OSTIM은 사전 학습된 모델 $\phi_\theta$를 고정(frozen)한 상태에서 특징 공간(Feature space) 상의 연산을 통해 작동하는 추론 전용(Inference-only) 방법이다. 전체 흐름은 다음과 같다: 특징 추출 $\rightarrow$ 중심 정규화(Center-normalization) $\rightarrow$ 프로토타입 생성 및 임시 할당 $\rightarrow$ InfoMax 기반의 프로토타입 최적화 $\rightarrow$ 최종 분류 및 Outlier score 산출.

### 주요 구성 요소 및 수식 설명

**1. 중심 정규화 및 유사도 측정**
특징 벡터 $z$를 더 효과적으로 처리하기 위해 중심 정규화 변환 $\psi_\mu$를 적용한다. 샘플 $z_i$와 클래스 프로토타입 $w_k$ 사이의 유사도 $l_{ik}$는 다음과 같이 정의된다.
$$l_{ik} = \text{sim}(z_i, w_k) = \langle \psi_\mu(z_i), \psi_\mu(w_k) \rangle, \quad \text{where } \psi_\mu(z) = \frac{z - \mu}{\|z - \mu\|_2}$$
여기서 $\mu$는 태스크 내의 모든 특징들의 평균값($\mu_{\text{Task}}$)을 주로 사용한다.

**2. Implicit Outlier Prototype**
외부 인스턴스를 구분하기 위해 $K+1$번째 클래스인 Outlier category를 도입한다. 이때 새로운 파라미터를 추가하는 대신, 기존 $K$개 Inlier 프로토타입들의 평균 방향과 정반대 방향을 Outlier prototype으로 정의한다. Outlier logit $l_{i, K+1}$은 다음과 같이 계산된다.
$$l_{i, K+1} = -\frac{1}{K} \sum_{k=1}^K l_{ik} = \left\langle \psi_\mu(z_i), -\frac{1}{K} \sum_{k=1}^K \psi_\mu(w_k) \right\rangle$$
이 수식은 기하학적으로 Inlier들의 중심에서 가장 먼 방향을 Outlier의 대표 방향으로 설정함으로써, 알려지지 않은 샘플들이 이 방향으로 수렴하도록 유도한다.

**3. 프로토타입 최적화 (Prototype Refinement)**
초기 프로토타입 $w_k$는 지원 집합의 중심점으로 설정되며, 이후 다음과 같은 Open-set 버전의 Transductive loss를 최소화하며 최적화된다.
$$\min_{w} CE - \hat{I}_\alpha$$
여기서 $CE$는 지원 집합에 대한 교차 엔트로피 손실이며, $\hat{I}_\alpha$는 정보 최대화 항으로 다음과 같이 구성된다.
$$-\hat{I}_\alpha = \sum_{k=1}^{K+1} \hat{p}_k \log \hat{p}_k - \alpha \frac{1}{|Q|} \sum_{i=1}^{|Q|} \sum_{k=1}^{K+1} p_{ik} \log p_{ik}$$

- **Marginal Entropy (첫 번째 항):** 모든 클래스에 샘플이 균등하게 배분되도록 하여 특정 클래스로 쏠리는 Trivial solution을 방지한다.
- **Conditional Entropy (두 번째 항):** 각 쿼리 샘플이 $K+1$개의 카테고리 중 하나에 매우 높은 확신을 가지고 할당되도록 강제한다. 이를 통해 Inlier는 해당 클래스로, Outlier는 $K+1$번째 카테고리로 명확히 구분되게 한다.

## 📊 Results

### 실험 설정

- **데이터셋:** mini-ImageNet, tiered-ImageNet (표준 FSC), CUB, Aircraft, Fungi (Cross-domain).
- **측정 지표:** Closed-set 분류 정확도(Acc), Outlier 탐지 성능(AUROC, AUPR), 그리고 90% Recall에서의 정밀도(Prec@0.9).
- **기준선(Baselines):** k-NN 기반 OOD 탐지, SimpleShot (Inductive), OpenMax, SnatcherF, TIM-GD (Transductive) 등.

### 주요 결과

1. **정량적 성능:** mini-ImageNet과 tiered-ImageNet 실험에서 OSTIM은 Outlier 탐지 지표(AUROC, AUPR)에서 기존의 Inductive 및 Transductive 방법들을 압도하였다. 특히 5-shot 설정에서도 Inductive 방법들보다 6-7% 높은 AUROC/AUPR를 기록하며 강력한 성능을 보였다.
2. **Trade-off 해결:** 기존 Transductive 방법(예: TIM)은 Closed-set 정확도는 높였으나 Outlier 탐지 성능을 크게 떨어뜨렸으나, OSTIM은 두 지표 모두에서 최상위권의 성능을 유지하며 최적의 균형점을 찾았다.
3. **Cross-domain 강건성:** tiered-ImageNet으로 학습된 모델을 CUB, Aircraft, Fungi 데이터셋에 적용한 결과, 도메인 시프트가 발생한 상황에서도 OSTIM은 Strong Baseline(SimpleShot + k-NN) 대비 일관된 성능 향상을 보였다.
4. **Model-agnosticity 검증:** ResNet-50, ViT-B/16, Mixer-B/16 등 서로 다른 아키텍처와 다양한 사전 학습 전략(Supervised, Self-supervised 등)이 적용된 모델들에 OSTIM을 적용했을 때, 하이퍼파라미터 수정 없이도 일관되게 성능이 향상됨을 확인하였다. 특히 ViT-B/16 모델과 결합했을 때 가장 높은 성능(Acc 76%, AUROC 82%)을 달성하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 FSOSR 문제의 핵심 어려움이 특징 공간에서 Novel class들의 분포가 Base class들에 비해 훨씬 흩어져 있고 불분명하다는 점(MIF 지표로 증명)에 있음을 밝혀냈다. OSTIM은 이러한 불확실성을 해결하기 위해 '전이 학습'과 '가상 Outlier 프로토타입'이라는 두 가지 장치를 결합하였다. 특히 모델의 파라미터를 직접 수정하지 않고 특징 공간에서의 프로토타입만을 최적화함으로써, 향후 더 강력한 백본 모델이 등장하더라도 그 혜택을 그대로 누릴 수 있는 구조적 유연성을 갖추었다.

### 한계 및 미해결 질문

전이 학습의 특성상, OSTIM은 쿼리 집합의 통계적 특성(예: 클래스 불균형, 샘플 수)에 민감할 가능성이 있다. 논문에서는 이를 언급하며, 실제 적용 시 전이 학습 기반 방법과 유도 학습(Inductive) 기반 방법 중 어느 것을 선택해야 할지에 대한 심도 있는 분석이 추가로 필요함을 시사하였다.

### 비판적 해석

Implicit prototype을 정의하는 방식($-\text{average of inliers}$)이 매우 단순하지만 효과적이라는 점이 인상적이다. 이는 고차원 특징 공간에서 Inlier들이 특정 방향으로 군집화되어 있다면, 그 반대 방향이 가장 가능성 높은 Outlier 영역이라는 직관에 근거한다. 다만, Outlier들의 분포가 단일한 방향이 아닌 다방향으로 퍼져 있는 복잡한 Open-set 환경에서도 이 단순한 가정이 유효할지는 추가적인 검증이 필요해 보인다.

## 📌 TL;DR

본 논문은 적은 샘플로 알려진 클래스를 분류하고 미지의 클래스를 탐지하는 **Few-Shot Open-Set Recognition (FSOSR)** 문제를 해결하기 위해, 모델 독립적인 추론 방법인 **OSTIM**을 제안한다. OSTIM은 가상의 **Implicit Outlier Prototype**을 설정하고 **정보 최대화(InfoMax)** 원리를 통해 프로토타입을 최적화함으로써, 분류 정확도 손실 없이 외부 인스턴스 탐지 능력을 획기적으로 높였다. 이 방법은 특정 모델의 재학습 없이 어떤 사전 학습된 모델에도 즉시 적용 가능하며, 특히 최신 Vision Transformer 등 고성능 아키텍처와 결합했을 때 그 효과가 극대화된다.
