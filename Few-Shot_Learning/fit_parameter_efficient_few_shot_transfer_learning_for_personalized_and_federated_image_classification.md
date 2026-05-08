# FIT: Parameter-Efficient Few-Shot Transfer Learning for Personalized and Federated Image Classification

Aliaksandra Shysheya, John Bronskill, Massimiliano Patacchiola, Sebastian Nowozin, Richard E. Turner (2023)

## 🧩 Problem to Solve

본 논문은 개인화(Personalization) 및 연합 학습(Federated Learning) 환경에서 필수적인 두 가지 요구 사항, 즉 **소량의 데이터로 학습 가능해야 한다는 점(Data-efficiency)**과 **통신 효율적인 분산 학습 프로토콜을 지원해야 한다는 점(Parameter-efficiency)**을 동시에 해결하고자 한다.

현대 딥러닝 시스템에서 모델을 특정 사용자의 요구에 맞게 개인화하거나, 프라이버시 보호를 위해 데이터를 중앙 서버로 전송하지 않는 연합 학습을 수행할 때, 업데이트해야 하는 파라미터의 수가 많으면 저장 공간의 부담이 크고 통신 비용이 기하급수적으로 증가한다. 특히 각 클라이언트가 보유한 데이터 양이 매우 적은 Few-shot 상황에서는 모델이 쉽게 과적합(Overfitting)되는 문제가 발생한다.

기존의 전이 학습(Transfer Learning) 방식, 특히 Big Transfer (BiT)와 같은 알고리즘은 높은 정확도를 보이지만 네트워크의 모든 파라미터를 업데이트해야 하므로 파라미터 효율성이 매우 낮다. 반면, 메타 학습(Meta-learning) 방식은 업데이트 파라미터 수를 줄일 수 있으나, 메타 학습 데이터셋과 성격이 다른 새로운 데이터셋에 적용했을 때 정확도가 크게 떨어진다는 한계가 있다. 따라서 본 논문의 목표는 **정확도를 희생하지 않으면서도 업데이트 파라미터 수를 획기적으로 줄인 Few-shot 전이 학습 프레임워크를 개발하는 것**이다.

## ✨ Key Contributions

본 논문은 전이 학습의 강력한 표현력과 메타 학습의 파라미터 효율성을 결합한 **FiLM Transfer (FIT)**를 제안한다. 핵심 설계 아이디어는 다음과 같다.

1. **Frozen Backbone with FiLM Adapters**: 대규모 데이터셋으로 사전 학습된 백본 네트워크의 가중치는 고정(Freeze)하고, 매우 적은 수의 파라미터만으로 네트워크를 적응시킬 수 있는 FiLM(Feature-wise Linear Modulation) 레이어를 전략적으로 배치하여 파라미터 효율성을 극대화한다.
2. **Naive Bayes Classifier Head**: 일반적인 선형 분류기(Linear Head) 대신, 데이터로부터 자동으로 설정 가능하며 학습 파라미터가 매우 적은 Gaussian Naive Bayes 분류기를 최종 레이어로 사용한다.
3. **Episodic Fine-tuning**: 메타 학습에서 영감을 받은 에피소드 기반 학습 프로토콜을 도입하여, Few-shot 상황에서도 과적합을 방지하고 최적의 성능을 낼 수 있도록 학습 절차를 구성한다.

## 📎 Related Works

본 연구는 크게 두 가지 기존 연구 흐름에서 영감을 얻었다.

첫째, **파라미터 효율적 어댑터(Parameter-efficient adapters)** 연구이다. Residual adapters를 비롯하여 LoRA, VPT, AdaptFormer 등 백본을 고정한 채 소량의 파라미터만 학습시키는 방법들이 제안되었다. FIT는 이 중 가장 파라미터 효율적이면서도 표현력이 뛰어난 FiLM 레이어를 채택하였다.

둘째, **메타 학습 및 메트릭 학습(Metric learning)** 연구이다. ProtoNets와 같은 방식은 클래스 간의 거리를 기반으로 분류를 수행하며, 에피소드 학습(Episodic training)을 통해 적은 데이터로도 일반화 능력을 높인다.

기존의 어댑터 기반 전이 학습 시스템들은 대부분 최종 레이어로 선형 분류기를 사용하며, 메타 학습 시스템들은 주로 메트릭 학습 헤드를 사용한다. FIT의 차별점은 **전이 학습의 맥락에서 강력한 Naive Bayes 메트릭 헤드를 도입하고, 이를 에피소드 기반 미세 조정(Episodic fine-tuning)으로 학습시킨다**는 점이다. 이는 특히 Low-shot 환경에서 기존의 배치 학습 기반 선형 헤드보다 훨씬 뛰어난 성능을 보임을 입증하였다.

## 🛠️ Methodology

### 전체 시스템 구조

FIT는 사전 학습된 고정 백본(Frozen Backbone), FiLM 어댑터 레이어, 그리고 Naive Bayes 헤드로 구성된다.

### 1. FIT Backbone (FiLM Layers)

백본 네트워크 $\theta$는 고정하며, 학습 가능한 FiLM 파라미터 $\psi = \{\gamma, \beta\}$를 추가한다. FiLM 레이어는 합성곱 레이어에서 발생하는 활성화 값 $a_{ij}$를 다음과 같이 선형적으로 변조(Modulation)한다.

$$\text{FiLM}(a_{ij}, \gamma_{ij}, \beta_{ij}) = \gamma_{ij} a_{ij} + \beta_{ij}$$

여기서 $\gamma$는 스케일(scale)을, $\beta$는 시프트(shift)를 담당하는 스칼라 값이다. 본 논문에서는 ResNetV2 블록의 중간 $3 \times 3$ 합성곱 레이어 이후와 백본의 맨 마지막 부분에 FiLM 레이어를 배치하였다. 이를 통해 전체 파라미터의 $0.05\%$ 미만만을 업데이트하면서도 다양한 데이터셋에 유연하게 적응할 수 있다.

### 2. FIT Head (Gaussian Naive Bayes)

최종 분류기로 Gaussian Naive Bayes(GNB)를 사용한다. 테스트 입력 $x^*$에 대한 클래스 $c$의 확률은 다음과 같이 정의된다.

$$p(y^*=c|b_{\theta,\psi}(x^*), \pi, \mu, \Sigma) = \frac{\pi_c \mathcal{N}(b_{\theta,\psi}(x^*) | \mu_c, \Sigma_c)}{\sum_{c'} \pi_{c'} \mathcal{N}(b_{\theta,\psi}(x^*) | \mu_{c'}, \Sigma_{c'})}$$

여기서 $\pi_c$는 클래스 사전 확률, $\mu_c$는 클래스 평균, $\Sigma_c$는 공분산 행렬이다. 본 논문은 공분산 $\Sigma$를 처리하는 세 가지 변형을 제안한다.

- **QDA (Quadratic Discriminant Analysis)**: 각 클래스마다 개별 공분산 행렬을 사용한다. 성능은 좋으나 파라미터 수가 매우 많다.
- **LDA (Linear Discriminant Analysis)**: 모든 클래스가 하나의 공분산 행렬을 공유한다. 파라미터 효율성이 매우 뛰어나며 QDA와 유사한 성능을 보인다.
- **ProtoNets**: 공분산을 단위 행렬($I$)로 가정하여, 단순히 클래스 평균과의 유클리드 거리를 측정한다.

LDA의 경우, 계산 과정을 최적화하여 각 클래스당 $\mu_c^T \Sigma_{LDA}^{-1}$ (차원 $d_b$)와 $\mu_c^T \Sigma_{LDA}^{-1} \mu_c$ (스칼라)만 저장하면 되므로, 업데이트 파라미터 수를 극도로 줄일 수 있다.

### 3. FIT Training (Episodic Fine-tuning)

단순한 배치 학습은 Few-shot 상황에서 과적합 문제를 일으키므로, 메타 학습의 **에피소드 학습(Episodic training)** 방식을 도입한다.

1. **데이터 분할**: 다운스트림 데이터셋 $D$를 $D_{train}$과 $D_{test}$로 분리한다.
2. **태스크 샘플링**: 매 반복(iteration)마다 서포트 세트 $D_\tau^S$ (헤드 설정용)와 쿼리 세트 $D_\tau^Q$ (파라미터 최적화용)로 구성된 태스크 $\tau$를 무작위로 샘플링한다.
3. **최적화**: 서포트 세트를 이용해 $\pi, \mu, \Sigma$를 계산하여 헤드를 구성하고, 쿼리 세트에 대해 최대 가능도(Maximum Likelihood)를 최적화하여 FiLM 파라미터 $\psi$와 공분산 가중치 $e$를 학습한다.

$$\hat{L}(\psi, e) = \sum_{\tau=1}^{T} \sum_{q=1}^{Q_\tau} \log p(y_{\tau^*q} | h_e(b_{\theta,\psi}(x_{\tau^*q})), \pi(D_\tau^S), \mu(D_\tau^S), \Sigma(D_\tau^S))$$

## 📊 Results

### 실험 설정

- **백본**: BiT-M-R50x1 (ImageNet-21K 사전 학습)
- **데이터셋**: CIFAR10, CIFAR100, Pets, Flowers, VTAB-1k (19개 데이터셋), ORBIT (개인화), CIFAR100 (연합 학습)
- **비교 대상**: BiT (Standard Transfer Learning)

### 주요 결과

1. **Few-shot 성능 (Low-shot)**:
    - 10-shot 이하의 환경에서 FIT-LDA는 BiT보다 높은 정확도를 보이며, 업데이트 파라미터 수는 BiT의 $1\%$ 미만($\approx 0.01\text{M}$ vs $23.5\text{M}$)이다.
2. **VTAB-1k 벤치마크**:
    - FIT-LDA는 BiT를 능가하며, 특히 EfficientNetV2-M 백본을 사용했을 때 전체 평균 정확도 $74.9\%$로 SOTA 성능을 달성하였다.
3. **모델 개인화 (ORBIT 데이터셋)**:
    - FIT-LDA는 기존의 메타 학습 방법(Simple CNAPs, ProtoNets)보다 우수한 비디오 정확도를 보였으며, 업데이트 파라미터 수는 BiT 대비 수천 배 적어 기기 저장 공간 및 전송 비용을 획기적으로 줄였다.
4. **연합 학습 (Federated Learning)**:
    - CIFAR100 데이터셋에서 BiT와 유사한 정확도를 유지하면서도 통신 비용을 극적으로 낮췄다. 60라운드 전체 통신 비용 기준, BiT가 $14\text{B}$개의 파라미터를 전송할 때 FIT는 단 $7\text{M}$개만 전송하였다.

## 🧠 Insights & Discussion

**강점 및 통찰**

- **과적합 방지**: FiLM 레이어는 학습해야 할 파라미터 수가 매우 적기 때문에, 데이터가 부족한 상황에서 전체 모델을 미세 조정(Fine-tuning)할 때보다 과적합을 훨씬 효과적으로 억제한다.
- **헤드의 중요성**: Low-shot 영역에서는 선형 헤드보다 Naive Bayes 기반의 메트릭 헤드가 훨씬 안정적이고 높은 성능을 낸다. 이는 데이터가 적을 때 단순한 통계적 모델이 복잡한 선형 모델보다 일반화 능력이 좋음을 시사한다.

**한계 및 비판적 해석**

- **계산 비용**: Naive Bayes 헤드(LDA, QDA)는 매 학습 반복마다 $d_b \times d_b$ 크기의 공분산 행렬을 역행렬(Invert) 계산해야 하므로, 선형 헤드보다 계산 비용이 높다.
- **데이터 성격의 영향**: 사전 학습 데이터(ImageNet)와 다운스트림 데이터의 성격이 매우 다른 경우(예: dSprites와 같은 구조적 데이터), FIT-LDA의 성능이 BiT에 비해 떨어지는 경향이 있다. 이는 FiLM 어댑터만으로는 극심한 도메인 차이를 극복하는 데 한계가 있을 수 있음을 의미한다.

## 📌 TL;DR

본 논문은 **고정된 백본 $\rightarrow$ FiLM 어댑터 $\rightarrow$ Naive Bayes 헤드**로 이어지는 효율적인 아키텍처와 **에피소드 기반 학습**을 결합한 **FIT**를 제안한다. FIT는 업데이트 파라미터를 $1\%$ 수준으로 줄이면서도 Few-shot 환경에서 기존 BiT보다 우수한 정확도를 달성하였으며, 특히 통신 비용이 치명적인 연합 학습과 개인화 시나리오에서 압도적인 효율성을 입증하였다. 이는 향후 거대 모델(Foundation Models)을 저전력/저사양 기기에 배포하고 적응시키는 연구에 중요한 기여를 할 것으로 보인다.
