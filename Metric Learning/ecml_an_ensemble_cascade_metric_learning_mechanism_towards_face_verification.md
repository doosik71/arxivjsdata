# ECML: An Ensemble Cascade Metric Learning Mechanism towards Face Verification

Fu Xiong, Yang Xiao, Zhiguo Cao, Yancheng Wang, Joey Tianyi Zhou and Jianxin Wu (2020)

## 🧩 Problem to Solve

본 논문은 얼굴 검증(Face Verification) 문제를 두 가지 클래스의 세밀한 시각적 인식(fine-grained visual recognition) 문제로 정의하고, 이를 해결하기 위한 거리 측정 학습(Metric Learning) 방법론을 제안한다. 얼굴 검증의 핵심 과제는 동일 인물 간의 변동성(intra-person variation)은 줄이고, 서로 다른 인물 간의 차이(inter-person difference)는 극대화하는 것이다.

기존의 Metric Learning 접근 방식은 크게 두 가지 한계를 가진다. 첫째, 얕은 학습(Shallow Learning) 기반의 선형 모델들은 모델의 표현력이 부족하여 학습 데이터의 특성을 충분히 반영하지 못하는 과소적합(Underfitting) 문제가 빈번하게 발생한다. 둘째, 최근의 딥 메트릭 러닝(Deep Metric Learning) 모델들은 강력한 적합 능력을 갖추고 있으나, 방대한 양의 학습 데이터가 필요하며 과적합(Overfitting)에 취약하고 계산 비용이 매우 높다는 단점이 있다.

따라서 본 논문의 목표는 과소적합과 과적합 사이의 적절한 절충점(tradeoff)을 찾아, 계산 효율적이면서도 강력한 판별력을 가진 새로운 메트릭 러닝 메커니즘을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같은 두 가지 설계 아이디어로 요약된다.

1. **Ensemble Cascade Metric Learning (ECML) 메커니즘**: 딥러닝의 계층적 구조에서 영감을 받아, 여러 단계의 메트릭 러닝을 직렬로 연결하는 Cascade 구조를 통해 과소적합 문제를 해결한다. 동시에, 각 단계에서 특징(feature)을 무작위 그룹으로 나누어 학습하는 앙상블(Ensemble) 방식을 도입하여, 특정 차원에 과도하게 적합되는 과적합 문제를 억제한다.
2. **Robust Mahalanobis Metric Learning (RMML)**: 기존의 KISSME나 XQDA와 같은 closed-form solution 기반 방법론들이 겪는 공분산 행렬의 역행렬 계산 실패(computation failure) 문제를 해결하기 위해, 역행렬 계산이 필요 없는 강건한 Mahalanobis 거리 학습 방법을 제안한다.

## 📎 Related Works

기존의 Metric Learning 연구는 크게 선형 패러다임(Linear paradigm)과 비선형 패러다임(Nonlinear model)으로 나뉜다. 선형 방식은 변환 행렬 $M$을 학습하여 거리 $d_{ij} = (x_i - x_j)^T M (x_i - x_j)$를 계산하며, 비선형 방식은 커널 기반의 접근법을 사용한다.

특히 Closed-form solution을 제공하여 확장성이 좋은 KISSME와 XQDA가 주목받았으나, 이들은 가우시안 분포 가정을 기반으로 공분산 행렬의 역행렬을 계산해야 한다. 하지만 얼굴 데이터의 경우 동일 인물 간의 상관관계가 매우 높아 행렬이 특이 행렬(singular matrix)이 될 가능성이 크며, 이로 인해 계산 실패가 발생하거나 이를 막기 위한 정규화 과정에서 추정 오차가 발생한다는 한계가 있다.

## 🛠️ Methodology

### 1. Cascade Metric Learning (과소적합 해결)

Cascade 구조는 총 $L+1$개의 계층적 학습 단계로 구성된다. 앞선 $L$개의 단계는 이전 단계의 특징을 새로운 특징 공간으로 매핑하며, 마지막 단계에서 최종 거리 메트릭을 생성한다.

- **특징 매핑**: $l$번째 단계에서 학습된 Mahalanobis 변환 행렬 $M^l$을 Cholesky 분해하여 $M^l = P^l (P^l)^T$로 나타내고, 이를 통해 특징을 다음과 같이 매핑한다.
  $$f^l_k = P^l f^{l-1}_k$$
- **Modified Cholesky Decomposition (MCD)**: $M^l$이 양의 정부호(positive definite)가 아닐 경우 Cholesky 분해가 불가능하다. 이를 해결하기 위해 Schur 분해를 통해 고윳값(eigenvalue)을 구한 뒤, 음수 고윳값을 0으로 설정하여 $\hat{M}^l$을 만들고 이를 분해하는 MCD 방식을 제안한다.
- **Square Root Normalization**: 특정 차원의 값이 지나치게 커져 과적합을 유발하는 것을 막기 위해 다음과 같은 정규화를 수행한다.
  $$\phi(f^l_k) = \text{sgn}(f^l_k) \sqrt{|f^l_k|}$$

### 2. Ensemble Metric Learning (과적합 해결)

Cascade 구조로 인한 과적합을 방지하기 위해, 각 단계(최종 단계 제외)에서 입력 특징 벡터를 $N$개의 겹치지 않는 그룹으로 무작위 셔플링(Random Shuffle)한다.

- 각 그룹별로 독립적인 메트릭 러닝 $\rightarrow$ MCD $\rightarrow$ Square Root Normalization 과정을 거치며, 최종적으로 모든 그룹의 결과물을 다시 결합(concatenate)하여 다음 단계의 입력으로 사용한다.
- 단계가 진행될수록 앙상블 그룹의 수 $N^l$을 점진적으로 줄여($N^l = 2^{L-l+1}$), 약한 메트릭들이 점차 강한 메트릭으로 융합되도록 설계하였다.

### 3. Robust Mahalanobis Metric Learning (RMML)

RMML은 가우시안 분포 가정 없이, 동일 인물 간의 거리(intra-class)는 원점에 가깝게, 서로 다른 인물 간의 거리(inter-class)는 원점에서 멀어지게 학습하는 것을 목표로 한다.

- **목적 함수**: 판별 항 $g_1$과 정규화 항 $g_2$의 가중합을 최소화한다.
  $$\hat{M} = \arg \min_M \lambda g_1 + g_2$$
  여기서 $g_1$은 동일 쌍과 상이 쌍의 정규화된 거리 차이이며, $g_2 = \frac{1}{2}\|M-I\|_F$는 특징 공간의 왜곡을 방지하는 Frobenius norm 정규화 항이다.
- **Closed-form Solution**: 위 문제는 볼록 최적화(convex optimization) 문제이며, 유도 과정을 통해 다음과 같은 closed-form solution을 얻는다.
  $$\hat{M} = I + \lambda \tilde{M}$$
  이 식은 역행렬 계산을 포함하지 않으므로 매우 안정적이며 계산 속도가 빠르다.

## 📊 Results

### 실험 설정

- **데이터셋**: MS-Celeb-1M (대규모 얼굴 데이터셋).
- **입력 특징**: CNN 특징(640차원) 및 Fisher Vector (FV) 특징.
- **평가 지표**: Equal Error Rate (EER, $\downarrow$).
- **비교 대상**: LMNN, LDML, ITML, KISSME, SILD, XQDA 등.

### 주요 결과

1. **CNN 특징 기반 성능**: EC-RMML이 모든 PCA 차원(640, 320, 160, 80, 40)에서 기존의 모든 선형 메트릭 러닝 방법을 압도하는 성능을 보였다. 특히 소규모 및 대규모 프로토콜 모두에서 일반화 능력이 뛰어남을 확인하였다.
2. **특징 분포에 따른 성능 차이**:
    - CNN 특징의 경우 RMML이 KISSME보다 성능이 좋은데, 이는 CNN 특징의 차이(difference) 분포가 가우시안 형태가 아니기 때문이다.
    - 반면 FV 특징의 경우 KISSME나 XQDA가 더 좋은 성능을 보였는데, FV 특징의 차이 분포는 가우시안 분포에 가깝기 때문으로 분석된다.
3. **ECML의 범용성**: ECML 메커니즘을 XQDA에 적용한 EC-XQDA 역시 원본 XQDA보다 향상된 성능을 보여, ECML이 특정 알고리즘에 국한되지 않고 다양한 메트릭 러닝 방법의 성능을 높일 수 있음을 입증하였다.
4. **효율성**: Closed-form solution을 사용하므로 학습 및 테스트 속도가 매우 빠르며, ECML 도입으로 인한 추가 시간 소모는 미미한 수준이다.

## 🧠 Insights & Discussion

본 논문은 메트릭 러닝에서 가장 고질적인 문제인 과소적합과 과적합의 균형을 Cascade 구조와 Ensemble 기법의 조합으로 해결하였다. 특히, 단순히 모델을 깊게 쌓는 것이 아니라 특징 셔플링을 통한 앙상블을 결합함으로써, 얕은 학습의 효율성과 깊은 학습의 표현력을 동시에 확보하였다.

또한, RMML을 통해 기존 Closed-form 방법론들의 치명적 약점이었던 수치적 불안정성(역행렬 계산 실패)을 제거하였다. 실험을 통해 도출된 중요한 통찰은 **"최적의 메트릭 러닝 방법은 입력 특징의 분포 특성에 따라 달라진다"**는 점이다. 이는 특정 알고리즘의 절대적 우위보다는 데이터의 통계적 특성을 먼저 파악하는 것이 중요함을 시사한다.

한계점으로는 Cascade 단계 수 $L$이 너무 커지면 다시 과적합이 발생한다는 점이 언급되었다. 저자들은 향후 연구에서 ResNet의 잔차 연결(Residual connection) 아이디어를 도입하여 더 깊은 Cascade 구조를 안정적으로 학습시키고자 한다.

## 📌 TL;DR

본 연구는 얼굴 검증을 위해 과소적합을 해결하는 **Cascade 구조**와 과적합을 억제하는 **Ensemble 그룹화**를 결합한 **ECML 메커니즘**과, 역행렬 계산 없이 안정적으로 작동하는 **RMML** 알고리즘을 제안하였다. 실험 결과, 특히 CNN 특징을 사용할 때 기존 최신 기법들보다 훨씬 낮은 EER을 기록하며 뛰어난 판별력과 일반화 성능, 계산 효율성을 입증하였다. 이 연구는 대규모 얼굴 인식 시스템에서 실시간성에 가까운 속도로 높은 정확도를 구현하는 데 중요한 기여를 할 가능성이 크다.
