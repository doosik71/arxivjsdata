# Adaptive Siamese Tracking with a Compact Latent Network

Xingping Dong, Jianbing Shen, Fatih Porikli, Jiebo Luo, and Ling Shao (2023)

## 🧩 Problem to Solve

본 논문은 Siamese 기반 트래커들이 유사한 방해 요소(similar distractors)가 존재하거나 대상의 변형(significant deformation)이 심한 까다로운 상황에서 추적에 실패하는 문제를 해결하고자 한다.

저자들은 이러한 실패의 근본 원인이 오프라인 학습 단계에서 **결정적인 샘플(decisive samples)**이 부족했기 때문이라는 점을 분석하였다. 즉, 학습 데이터셋에 포함되지 않은 특이 케이스들이 테스트 시에 나타나면, 모델의 판별 능력이 떨어져 잘못된 대상(negative sample)을 타겟으로 오인하게 된다. 특히, 대부분의 Siamese 트래커들이 첫 번째 프레임에서 제공되는 풍부한 시퀀스 특화 정보(sequence-specific information)를 템플릿 추출에만 사용하고, 이를 모델의 판별 능력을 높이는 컨텍스트 정보로 활용하지 않는다는 점을 지적한다.

따라서 본 논문의 목표는 첫 번째 프레임의 정보를 효율적으로 이용하여 베이스 모델을 빠르게 조정(adapt)함으로써, 시퀀스별 맞춤형 판별 능력을 갖춘 트래커를 구현하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Siamese 트래킹 과제를 **이진 분류(binary classification)** 문제로 단순화하여 해석하고, 첫 번째 프레임의 샘플들을 통해 모델의 결정 경계(decision hyperplane)를 최적화하는 것이다.

1. **CLNet (Compact Latent Network) 제안**: 통계 기반의 잠재 특징(statistics-based latent features)을 사용하여 베이스 모델의 마지막 레이어 가중치를 동적으로 조정하는 경량 네트워크를 제안하였다.
2. **통계 기반 잠재 특징 활용**: 샘플들의 평균($\mu$)과 표준편차($\sigma$)를 추출하여 가우시안 분포로 근사함으로써, 결정적인 샘플이 부족하더라도 강건한 판별 경계를 찾을 수 있도록 설계하였다.
3. **Diverse Sample Mining 전략**: 학습 시 단순히 IoU 기반으로 샘플을 뽑는 것이 아니라, 기존 모델이 헷갈려 하는(분류 점수가 높은) 부정 샘플을 채굴하여 CLNet의 판별 능력을 극대화하였다.
4. **Conditional Updating (CU) 전략**: 모든 프레임에서 업데이트를 수행하는 대신, 신뢰도 점수와 마진(margin)을 기준으로 장면 변화가 감지될 때만 선택적으로 모델을 업데이트하여 연산 효율성과 정확도를 동시에 잡았다.

## 📎 Related Works

### Siamese Networks Based Trackers

SiamFC, SiamRPN, SiamRPN++ 등으로 대표되는 Siamese 네트워크들은 오프라인 학습과 온라인 추적을 분리하여 매우 빠른 속도를 자랑한다. 그러나 이들은 고정된 가중치를 사용하므로, 특정 비디오 시퀀스 내에서만 발생하는 특수한 배경이나 방해 요소에 대응하는 능력이 부족하다는 한계가 있다.

### Meta-Learning

최근 Meta-learning을 이용해 빠르게 적응(fast adaptation)하는 연구들이 진행되었으나, 대부분의 최적화 기반(optimization-based) 방법들은 온라인 학습 시 반복적인 경사 하강법(SGD)을 수행해야 하므로 실시간 트래킹에 적용하기에는 연산 비용이 너무 크다는 단점이 있다.

본 연구는 이러한 기존 방식들과 달리, 통계 기반의 매우 작은 잠재 네트워크(CLNet)를 통해 가중치 편차($\Delta\theta$)를 직접 예측함으로써 실시간성을 유지하면서도 적응 능력을 확보하였다.

## 🛠️ Methodology

### 전체 파이프라인

CLNet은 베이스 트래커(SiamRPN++, SiamFC, SiamBAN)의 마지막 레이어 가중치를 조정하는 보조 네트워크이다. 전체 구조는 **특징 조정 서브네트워크 $\rightarrow$ 잠재 인코더 $\rightarrow$ 예측 서브네트워크** 순으로 구성된다.

### 주요 구성 요소 및 절차

**1. 특징 조정 서브네트워크 (Feature-adjusting Subnetwork)**
베이스 모델에서 생성된 특징 맵 $M \in \mathbb{R}^{w \times h \times c}$를 입력으로 받아 세 개의 $1\times1$ 컨볼루션 레이어를 통해 차원을 조정하여 $\bar{M} \in \mathbb{R}^{w \times h \times \bar{c}}$를 생성한다.

**2. 잠재 인코더 (Latent Encoder)**
조정된 특징 맵 $\bar{M}$을 정답 라벨 $Y$에 따라 긍정 집합($P$)과 부정 집합($N$)으로 나눈다. 각 집합의 평균 $\mu$와 표준편차 $\sigma$를 계산하여 통계 기반의 컴팩트한 특징 $c$를 생성한다.
$$\mu_{\rho} = \frac{1}{n_{\rho}} \sum_{i=1}^{n_{\rho}} \bar{m}_{\rho i}, \quad \sigma_{\rho} = \sqrt{\frac{1}{n_{\rho}} \sum_{i=1}^{n_{\rho}} (\bar{m}_{\rho i} - \mu_{\rho})^2} \quad (\rho \in \{+, -\})$$
최종 잠재 특징은 다음과 같이 결합된다: $c = \text{concat}(\mu^+, \sigma^+, \mu^-, \sigma^-)$.

**3. 예측 서브네트워크 및 가중치 업데이트**
MLP(Multi-Layer Perceptron)를 통해 가중치 편차 $\Delta\theta_1$을 예측하고, 이를 기존 가중치 $\theta_1$에 더해 조정된 가중치 $\theta_a$를 생성한다.
$$\theta_a = \theta_1 + \Delta\theta_1$$
이 조정된 가중치는 분류(classification) 및 회귀(regression) 브랜치의 마지막 레이어에 적용된다.

### 학습 전략 및 추론 절차

- **Diverse Sample Mining**: 학습 시 한 배치를 동일 시퀀스에서 샘플링하며, 기존 모델의 분류 점수가 높은(즉, 타겟과 매우 유사하게 생긴) 부정 샘플들을 추가로 채굴하여 학습에 활용함으로써 결정 경계를 더 정교하게 만든다.
- **Conditional Updating (CU)**:
  - **신뢰도 확인**: 예측된 박스의 점수 $s_i$가 임계값 $\tau_r$보다 높을 때만 해당 프레임을 업데이트를 위한 신뢰할 수 있는 샘플로 간주한다.
  - **업데이트 트리거**: 긍정 샘플의 최고점과 부정 샘플의 최고점 차이인 마진 $\eta_i = s^*_{p,i} - s^*_{n,i}$가 $\tau_m$보다 작아지면(즉, 판별이 어려워지면) 모델을 업데이트한다.

## 📊 Results

### 실험 설정

- **대상 모델**: SiamRPN++, SiamFC, SiamBAN
- **데이터셋**: NfS, DTB, LaSOT, GOT10k, VOT2019, VOT2020
- **지표**: Precision, AUC, EAO, ACC, ROB 등

### 주요 결과

1. **정량적 성능 향상**: 모든 베이스 모델에서 CLNet 적용 후 성능이 크게 향상되었다. 특히 CLNet*-BAN은 NfS30 데이터셋에서 기존 최상위 모델인 MDNet보다 Precision과 AUC 면에서 각각 29.8%, 31.1% 높은 성과를 보였다.
2. **실시간성 유지**: 추가된 CLNet의 파라미터 수가 매우 적어 연산 오버헤드가 거의 없다. 조정된 트래커들의 속도는 38 FPS ~ 104.9 FPS 범위를 유지하며 실시간 동작이 가능하다.
3. **범용성 검증**: 서로 다른 구조를 가진 세 가지 트래커(SiamRPN++, SiamFC, SiamBAN) 모두에 적용하여 일관된 성능 향상을 이끌어냄으로써 제안 방법의 일반화 능력을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구의 가장 큰 통찰은 **통계 기반 특징(statistics-based features)**의 활용이다. 개별 샘플 값만으로는 결정적인 샘플이 누락되었을 때 잘못된 결정 경계를 설정할 위험이 크지만, 평균과 표준편차를 통해 데이터의 분포(Gaussian distribution)를 근사하면 샘플 수가 적더라도 보다 이상적인 결정 경계를 찾을 수 있음을 수학적/시각적으로 증명하였다.

### 한계 및 비판적 논의

1. **Transformer 모델과의 격차**: 최신 Transformer 기반 트래커(TransT, STARK 등)와 비교했을 때 AUC 성능 차이가 존재한다. 이는 CLNet의 문제라기보다 베이스 모델로 사용된 ResNet 기반 백본의 표현력 한계에서 기인한 것으로 보인다.
2. **가중치 업데이트 방식**: CBAM이나 FILM 같은 복잡한 가중치 증강 기법보다 단순한 덧셈(additive augmentation) 방식이 더 효과적이었다는 결과는 흥미롭지만, 왜 단순 덧셈이 더 우수한지에 대한 더 깊은 이론적 분석이 보완될 필요가 있다.

## 📌 TL;DR

이 논문은 Siamese 트래커가 특정 상황에서 실패하는 원인을 '결정적 샘플의 부족'으로 정의하고, 이를 해결하기 위해 첫 프레임의 정보를 이용해 모델을 빠르게 조정하는 **CLNet**을 제안하였다. 통계 기반의 잠재 특징을 통해 연산량 증가 없이 판별 능력을 극대화했으며, 다양한 Siamese 트래커에 적용 가능함을 보였다. 향후 더 강력한 백본(예: Transformer)에 이 구조를 결합한다면 더욱 강력한 실시간 적응형 트래커가 될 가능성이 높다.
