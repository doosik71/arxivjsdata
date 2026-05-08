# TAnoGAN: Time Series Anomaly Detection with Generative Adversarial Networks

Md Abul Bashar, Richi Nayak

## 🧩 Problem to Solve

시계열 데이터에서 이상(Anomaly) 탐지는 제조, 의료 영상, 사이버 보안 등 다양한 분야에서 중요한 문제로 다루어집니다. 특히, 레이블 정보가 부족하여 비지도 학습으로 접근하는 경우가 많으며, 기존의 선형 모델은 시계열 데이터의 비선형 상호작용을 처리하지 못하고, 동적인 시스템에서는 '정상 범위'를 정의하기 어렵다는 한계가 있습니다. 또한, 신경망 기반 모델들은 일반적으로 많은 수의 데이터 포인트에서 훈련되어야 하지만, 실제 시계열 데이터셋은 소량의 데이터만 사용 가능한 경우가 많아 이러한 소규모 데이터셋에서 효과적인 이상 탐지 방법론을 개발하는 것이 핵심 문제입니다.

## ✨ Key Contributions

- **소규모 데이터셋을 위한 GAN 기반 이상 탐지 모델 제안:** 소수의 데이터 포인트만 사용할 수 있는 시계열 데이터셋에서 이상을 탐지하기 위한 새로운 GAN 기반 비지도 학습 방법인 TAnoGan을 제안했습니다.
- **광범위하고 다양한 데이터셋을 통한 성능 검증:** NAB(Numenta Anomaly Benchmark) 컬렉션의 46개 실제 시계열 데이터셋에 대해 TAnoGan을 평가하여, 다양한 도메인과 소규모 데이터셋에서의 일반화 성능을 입증했습니다.
- **LSTM 기반 GAN의 우수성 입증:** LSTM 기반 GAN이 전통적인 LSTM 모델 대비 적대적 훈련(adversarial training)을 통해 성능을 향상시킬 수 있음을 보였습니다.
- **최첨단 모델 대비 뛰어난 성능:** TAnoGan이 기존 전통 모델 및 최첨단 신경망 기반 모델들보다 더 나은 이상 탐지 성능을 보임을 입증했습니다.

## 📎 Related Works

- **전통적인 비지도 이상 탐지 방법:**
  - **선형 모델:** PCA(Principal Component Analysis), PLS(Partial Least Squares)는 가우시안 분포를 가정하고 상관관계가 높은 데이터에만 효과적입니다.
  - **거리 기반 모델:** KNN(K-Nearest Neighbor)은 이상 지속 시간이나 이상 개수에 대한 사전 지식을 요구합니다.
  - **밀도 기반/확률 모델:** ABOD(Angle-Based Outlier Detection), FB(Feature Bagging), GMM(Gaussian Mixture Model), IsoF(Isolation Forest), OCSvm(One Class Support Vector Machine) 등이 있으나, 시계열 데이터의 시간적 상관관계를 고려하지 못하거나 비선형 패턴 학습에 한계가 있습니다.
- **딥러닝 기반 비지도 이상 탐지 방법:** Auto-Encoder, Encoder-Decoder, LSTM과 같은 모델들이 유망한 성능으로 인기를 얻고 있습니다.
- **GAN 기반 이상 탐지:**
  - 대부분 이미지 도메인(예: AnoGan)에서 연구되었으며, CNN을 사용합니다.
  - 시계열 도메인에서는 데이터 생성(예: Recurrent Conditional GAN)에 주로 사용되거나, MAD-GAN과 같이 대규모 데이터셋에 효과적인 것으로 알려져 있습니다.
  - TAnoGan은 GAN을 시계열 이상 탐지에 적용하고, 특히 소규모 데이터셋 문제를 해결하기 위해 LSTM을 Generator와 Discriminator에 사용하는 것이 특징입니다.

## 🛠️ Methodology

TAnoGan은 크게 두 가지 하위 프로세스로 구성됩니다:

1. **정상 데이터 분포 학습:**
   - **GAN 구성:** Generator ($G$)와 Discriminator ($D$)를 모두 LSTM 네트워크로 구현합니다.
   - **소규모 데이터셋 아키텍처:** 소규모 데이터셋에 대한 과적합을 방지하고 사실적인 데이터 생성을 위해, 깊이가 얕은 $D$ (1개 LSTM 레이어, 100개 히든 유닛)와 중간 깊이의 $G$ (3개 스택형 LSTM 레이어: 32, 64, 128개 히든 유닛)를 사용합니다. $G$의 각 레이어에서 히든 유닛 수를 점진적으로 늘려 계층적 디코딩을 가능하게 합니다.
   - **훈련:** $G$는 잠재 공간 $Z$에서 샘플링된 노이즈 벡터 $z$로부터 가짜 시계열 시퀀스를 생성하고, $D$는 실제 시계열 시퀀스와 $G$가 생성한 가짜 시퀀스를 구별하도록 훈련됩니다. 훈련은 $G$와 $D$가 경쟁적으로 최적화하는 미니맥스 게임(minimax game) 형태로 진행됩니다.
   - **목적 함수:** $V(D,G) = E_{x \sim p_{data}(x)}[\log D(x)] + E_{z \sim p_{z}(z)}[\log(1-D(G(z)))]$를 사용하여 $D$는 $D(x)$를 최대화하고 $D(G(z))$를 최소화하며, $G$는 $D(G(z))$를 최대화합니다.
   - **데이터 전처리:** 원본 시계열 데이터를 슬라이딩 윈도우 $s_w$를 사용하여 작은 시퀀스로 나눕니다.
2. **실제 데이터를 잠재 공간에 매핑 및 이상 탐지:**
   - **역매핑:** GAN은 $G: Z \to X$ 매핑을 학습하지만, $G^{-1}: X \to Z$와 같은 역매핑은 제공하지 않습니다.
   - **최적의 $z$ 탐색:** 주어진 실제 시퀀스 $x$에 대해, $x$와 가장 유사한 가짜 시퀀스 $G(z)$를 생성하는 최적의 $z$를 찾기 위해 반복적인 경사 하강법을 사용합니다 (이때 $G$와 $D$의 파라미터는 고정).
   - **손실 함수 ($L$):**
     - **잔차 손실 ($L_R$):** 실제 시퀀스 $x$와 재구성된 가짜 시퀀스 $G(z_\lambda)$ 간의 지점별 불일치를 측정합니다. $L_R(z_\lambda) = \sum |x-G(z_\lambda)|$.
     - **식별 손실 ($L_D$):** Discriminator의 중간 계층 특징 표현 $f(\cdot)$을 사용하여 $f(x)$와 $f(G(z_\lambda))$ 간의 불일치를 측정합니다. $L_D(z_\lambda) = \sum |f(x)-f(G(z_\lambda))|$.
     - **총 손실:** $L(z_\lambda) = (1-\gamma) \cdot L_R(z_\lambda) + \gamma \cdot L_D(z_\lambda)$ ($\gamma$는 가중치 계수).
   - **이상 점수 ($A(x)$) 계산:** 최종 $\Lambda$번째 업데이트 반복에서의 $L_R(z_\Lambda)$와 $L_D(z_\Lambda)$로부터 이상 점수 $A(x) = (1-\gamma) \cdot R(x) + \gamma \cdot D(x)$를 도출합니다. $A(x)$가 높을수록 이상치일 가능성이 큽니다.

## 📊 Results

- **데이터셋:** NAB 컬렉션의 46개 실제 시계열 데이터셋(각 1천~2만 2천 데이터 포인트)을 사용했습니다.
- **평가 지표:** Accuracy, Precision, Recall, F1 Score, Cohen Kappa, AUC를 사용했습니다.
- **기준 모델:** MadGan, AutoEncoder, VanLstm, Isolation Forest, GMM, OCSvm, BiLstmGan, CnnGan과 비교했습니다.
- **주요 결과:**
  - **누적 랭킹:** TAnoGan은 F1 Score, Cohen Kappa Score, Precision, Accuracy, AUC에서 모든 기준 모델보다 높은 랭킹을 기록했습니다 (Figure 3).
  - **쌍별 비교:** 대부분의 데이터셋에서 TAnoGan이 모든 기준 모델보다 우수한 성능을 보였습니다 (Figure 4).
  - **데이터셋 크기 및 도메인:** TAnoGan은 5000개 이상의 인스턴스를 가진 데이터셋에서 항상 대부분의 모델보다 우수했으며, 5000개 미만의 작은 데이터셋에서도 대부분의 경우 좋은 성능을 보였습니다. 데이터셋의 도메인에는 민감하지 않았습니다 (Figure 5).
  - **임계 차이(Critical Difference) 다이어그램:** TAnoGan이 가장 낮은 점수를 기록하여 모든 모델 중 가장 우수한 성능을 나타냈습니다 (Figure 6). Recall 지표에서는 GMM이 미세하게 좋았으나, GMM은 많은 오탐지(false positive)로 인해 Precision과 F1 Score가 낮았습니다. TAnoGan은 Recall과 Precision의 균형(높은 F1 Score)을 달성했습니다.

## 🧠 Insights & Discussion

- **GAN의 효과:** TAnoGan의 결과는 시계열 데이터 이상 탐지에서 GAN이 데이터의 일반적인 분포를 학습하여 이상치를 효과적으로 분리하는 데 유용함을 입증합니다.
- **적대적 훈련의 이점:** TAnoGan이 VanLstm 및 AutoEncoder와 같은 딥러닝 모델보다 우수한 성능을 보인 것은 적대적 훈련이 데이터의 일반적인 분포를 더 잘 학습하게 함을 시사합니다.
- **비선형 패턴 학습 능력:** OCSvm이나 IsoF와 같은 모델을 능가하는 TAnoGan의 성능은 비선형 패턴을 효과적으로 학습하는 LSTM 기반 GAN의 능력을 보여줍니다.
- **Generator 아키텍처의 중요성:** Generator의 LSTM 레이어에서 점진적으로 증가하는 히든 유닛(32, 64, 128)은 시퀀스의 세분화된(fine-grained) 속성과 거친(coarse-grained) 속성을 계층적으로 디코딩하여, 작은 데이터셋에서도 일반화 성능을 향상시키는 데 기여했습니다. 이는 MAD-GAN, BiLstmGan, CnnGan과 같은 유사 GAN 아키텍처 모델들보다 TAnoGan이 우수한 이유입니다.
- **한계 및 향후 연구:** 최적의 윈도우 길이 결정, 모델 불안정성, 에포크 수에 대한 민감성 등 GAN 기반 이상 탐지의 알려진 문제점들이 있으며, 향후 대규모 데이터셋 및 다양한 아키텍처에 대한 추가 연구가 필요합니다.

## 📌 TL;DR

TAnoGan은 소규모 시계열 데이터셋의 이상 탐지를 위한 새로운 GAN 기반 비지도 학습 모델입니다. Generator와 Discriminator에 LSTM을 사용하며, Generator는 계층적 디코딩을 위해 점진적으로 히든 유닛이 증가하는 구조를 채택했습니다. 46개 NAB 실제 데이터셋에 대한 광범위한 평가 결과, TAnoGan은 기존의 전통 및 신경망 모델들보다 일관되게 우수한 성능을 보이며, 특히 작은 데이터셋에서 적대적 훈련을 통한 일반 데이터 분포 학습의 효과를 입증했습니다.
