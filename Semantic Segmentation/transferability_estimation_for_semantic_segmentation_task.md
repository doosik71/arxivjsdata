# Transferability Estimation for Semantic Segmentation Task

Yang Tan, Yang Li, Shao-Lun Huang (2024)

## 🧩 Problem to Solve

본 논문은 전이 학습(Transfer Learning)에서 소스 모델(Source Model)이나 소스 태스크(Source Task)를 타겟 태스크(Target Task)로 전이했을 때, 그 성능이 얼마나 좋을지를 미리 예측하는 **전이 가능성 추정(Transferability Estimation)** 문제를 다룬다.

전이 가능성 점수(Transferability Score)를 통해 실제 전이 학습을 수행하지 않고도 효율적으로 전이 가능성이 높은 소스 모델을 선택할 수 있다는 점에서 매우 중요하다. 기존의 분석적 전이 가능성 지표들은 주로 이미지 분류(Image Classification) 문제에 설계되어 있었으며, 자율 주행이나 의료 영상 분석의 핵심인 **시맨틱 세그멘테이션(Semantic Segmentation)** 태스크에 대한 전이 가능성 추정 연구는 부족한 실정이다.

시맨틱 세그멘테이션은 픽셀 단위의 분류 문제로 볼 수 있으나, 출력 차원이 매우 높기 때문에 기존의 분류용 지표를 그대로 적용하기 어렵다. 따라서 본 논문의 목표는 기존의 분석적 지표인 **OTCE (Optimal Transport based Conditional Entropy)** 점수를 시맨틱 세그멘테이션 태스크로 확장하여 효율적인 전이 가능성 추정 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시맨틱 세그멘테이션 모델의 방대한 출력 데이터(픽셀 단위)로 인해 발생하는 계산 비용 문제를 해결하기 위해, **무작위 픽셀 샘플링(Random Pixel Sampling)** 전략을 도입하여 OTCE 점수를 계산하는 것이다.

구체적으로, 전체 픽셀을 대상으로 최적 운송(Optimal Transport) 문제를 푸는 대신, 소스와 타겟 데이터셋에서 각각 $N$개의 픽셀을 무작위로 샘플링하여 OTCE를 계산하고, 이를 $K$번 반복하여 평균을 내는 방식을 통해 계산 복잡도를 획기적으로 줄이면서도 전이 가능성을 정확하게 예측하고자 하였다.

## 📎 Related Works

전이 학습은 레이블이 적은 태스크의 성능을 향상시키기 위해 관련 소스 태스크의 지식을 활용하는 유용한 방법이다. 특히 시맨틱 세그멘테이션은 픽셀 단위의 정밀한 수동 레이블링 비용이 매우 높기 때문에 전이 학습의 필요성이 매우 크다.

최근 LEEP나 OTCE와 같은 분석적 전이 가능성 지표(Analytical Transferability Metrics)들이 제안되었으나, 이들은 대부분 이미지 레벨의 분류 문제를 대상으로 한다. 시맨틱 세그멘테이션의 경우, 데이터의 단위가 이미지가 아닌 픽셀이 되므로 샘플 사이즈가 급격히 증가한다(예: $10^4$에서 $10^7$ 수준으로 증가). 이로 인해 기존의 OTCE 프레임워크를 그대로 적용할 경우 제한된 계산 자원(예: 개인용 컴퓨터)으로는 최적 결합(Optimal Coupling) 행렬을 찾는 것이 불가능하다는 한계가 있다.

## 🛠️ Methodology

### 1. 전이 가능성 정의 (Transferability Problem)

소스 데이터셋 $D_s = \{(x_i^s, y_i^s)\}_{i=1}^m$와 타겟 데이터셋 $D_t = \{(x_i^t, y_i^t)\}_{i=1}^n$이 있을 때, 소스 모델 $\theta_s$를 타겟 모델 $\theta_t$로 파인튜닝(Finetune)한다. 이때의 경험적 전이 가능성(Empirical Transferability)은 타겟 테스트 세트에서의 기대 로그 가능도(Expected log-likelihood)로 정의된다.

$$Trf(S \to T) = \mathbb{E}[\log P(y_t | x_t; \theta_t)]$$

### 2. 시맨틱 세그멘테이션을 위한 OTCE 점수

OTCE 점수는 도메인 차이(Domain Difference)와 태스크 차이(Task Difference)의 선형 결합으로 전이 가능성을 설명한다. 본 논문에서는 구현의 실용성을 위해 **태스크 차이($W^T$)**만을 사용하여 전이 가능성을 설명한다.

**계산 절차는 다음과 같다:**

1. **특징 추출:** 소스 모델 $\theta_s$를 사용하여 소스 및 타겟 데이터셋에서 특징 맵(Feature maps)을 추출한다.
2. **픽셀 데이터셋 구축:** 각 픽셀의 특징 벡터 $f$와 레이블 $y$를 쌍으로 하여 픽셀 단위 데이터셋 $D_{pix}^s$와 $D_{pix}^t$를 구성한다.
3. **최적 운송(Optimal Transport) 문제 해결:** 엔트로피 정규화(Entropic Regularizer)가 포함된 OT 문제를 풀어 최적 결합 행렬 $\pi^*$를 구한다.

$$\text{OT}(D_{pix}^s, D_{pix}^t) \triangleq \min_{\pi \in \Pi} \sum_{i,j=1}^{N_s, N_t} c(f_i^s, f_j^t)\pi_{ij} + \epsilon H(\pi)$$

여기서 $c(\cdot, \cdot) = \|\cdot - \cdot\|_2^2$는 비용 함수이며, Sinkhorn 알고리즘을 통해 $\pi^*$를 효율적으로 계산한다.
4.  **결합 확률 분포 계산:** $\pi^*$를 이용하여 소스와 타겟 레이블 간의 경험적 결합 확률 분포 $\hat{P}(y_s, y_t)$와 소스 레이블의 주변 확률 분포 $\hat{P}(y_s)$를 계산한다.

$$\hat{P}(y_s, y_t) = \sum_{i,j: y_i^s=y_s, y_j^t=y_t} \pi_{ij}^*$$
$$\hat{P}(y_s) = \sum_{y_t \in Y_t} \hat{P}(y_s, y_t)$$

1. **조건부 엔트로피(Conditional Entropy) 계산:** 태스크 차이 $W^T$를 조건부 엔트로피로 계산한다.

$$W^T = H(Y_t | Y_s) = -\sum_{y_t \in Y_t} \sum_{y_s \in Y_s} \hat{P}(y_s, y_t) \log \frac{\hat{P}(y_s, y_t)}{\hat{P}(y_s)}$$

최종적으로 전이 가능성 점수는 태스크 차이의 음수 값으로 정의된다: $\text{OTCE} = -W^T$.

### 3. 구현 및 샘플링 전략

모든 픽셀을 계산하는 대신, 다음과 같은 알고리즘을 통해 계산 비용을 줄인다.

- 소스와 타겟 픽셀 세트에서 각각 $N=10,000$개의 픽셀을 무작위로 샘플링한다.
- 이 과정을 $K=10$번 반복하여 OTCE 점수를 계산하고, 그 평균값을 최종 점수로 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Cityscapes, BDD100K, GTA5.
- **평가 지표:** 전이 정확도(Transfer Accuracy)와 OTCE 점수 간의 **피어슨 상관 계수(Pearson Correlation Coefficient)**.
- **전이 설정:**
  - **Intra-dataset:** Cityscapes 내의 서로 다른 도시 간 전이 (UNet 구조 사용, 타겟 데이터 20장으로 파인튜닝).
  - **Inter-dataset:** BDD100K 및 GTA5에서 학습된 6가지 모델(Fcn8s, UNet, SegNet, PspNet, FrrnA, FrrnB)을 Cityscapes로 전이.

### 2. 주요 결과

- **Intra-dataset 전이:** 특정 도시를 타겟으로 했을 때, OTCE 점수가 실제 전이 성능과 매우 높은 상관관계(최대 0.768)를 보였다. 이는 OTCE 점수를 통해 전이 가능한 소스 태스크를 신뢰성 있게 선택할 수 있음을 의미한다.
- **Inter-dataset 전이:**
  - BDD100K $\to$ Cityscapes 및 GTA5 $\to$ Cityscapes 설정 모두에서 OTCE 점수가 전이 성능의 좋은 지표가 됨을 확인하였다.
  - 특히, 실제 데이터인 BDD100K에서 전이한 모델이 가상 데이터인 GTA5에서 전이한 모델보다 더 높은 정확도를 보였으며, 이는 도메인 간의 격차(Domain Gap)가 클수록 전이 성능이 저하됨을 시사한다.

## 🧠 Insights & Discussion

본 연구는 고차원 출력 값을 갖는 시맨틱 세그멘테이션 태스크에 OTCE 지표를 성공적으로 확장 적용하였다. 특히, 단순한 무작위 샘플링과 반복 평균이라는 전략만으로도 계산 복잡도 문제를 해결하고 실제 전이 성능과 높은 상관관계를 유지했다는 점이 강점이다.

다만, 논문에서 언급되었듯이 원래의 OTCE는 도메인 차이($\hat{W}_D$)와 태스크 차이($\hat{W}_T$)를 모두 고려하는 구조이지만, 본 구현에서는 실용성을 위해 태스크 차이만을 사용하였다. 이는 도메인 차이가 전이 성능에 미치는 영향력을 완전히 반영하지 못했을 가능성이 있다. 하지만 인터-데이터셋 실험 결과에서 BDD100K와 GTA5의 성능 차이가 나타난 점을 볼 때, 태스크 차이 지표만으로도 상당 부분의 전이 가능성을 예측할 수 있음을 보여준다.

## 📌 TL;DR

본 논문은 이미지 분류용 전이 가능성 지표인 **OTCE를 시맨틱 세그멘테이션 태스크로 확장**한 연구이다. 픽셀 단위의 방대한 데이터를 처리하기 위해 **무작위 픽셀 샘플링 전략**을 제안하였으며, 실험을 통해 제안한 OTCE 점수가 실제 전이 성능과 높은 상관관계를 가짐을 입증하였다. 이 연구는 향후 시맨틱 세그멘테이션 모델 학습 시 최적의 사전 학습 모델이나 소스 데이터를 효율적으로 선택하는 데 기여할 수 있을 것으로 보인다.
