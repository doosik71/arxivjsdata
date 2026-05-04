# ALGAN: Time Series Anomaly Detection with Adjusted-LSTM GAN

Md Abul Bashar, Richi Nayak (2023)

## 🧩 Problem to Solve

본 논문은 시계열 데이터에서 정상적인 동작 범위를 벗어난 지점을 식별하는 비지도 학습(Unsupervised Learning) 기반의 이상치 탐지(Anomaly Detection) 문제를 해결하고자 한다. 시계열 데이터 분석은 제조, 의료 영상, 사이버 보안 등 다양한 도메인에서 매우 중요하며, 특히 미션 크리티컬한 환경에서는 치명적인 결과가 발생하기 전에 선제적으로 대응하는 것이 필수적이다.

기존의 접근 방식들은 몇 가지 한계를 가지고 있다. PCA나 PLS와 같은 선형 투영 및 변환 기법들은 시계열 데이터 내에 존재하는 비선형적 상호작용을 처리하는 데 한계가 있으며, 단순 예측 범위와 실제 값을 비교하는 방식은 시스템이 매우 동적으로 변화할 때 정상 범위를 정의하기 어렵다는 단점이 있다. 최근에는 Generative Adversarial Networks (GANs)가 복잡한 고차원 분포를 모델링하는 데 효과적임이 밝혀졌으나, 시계열 데이터의 핵심인 시간적 의존성(Temporal Dependencies)을 어떻게 효과적으로 캡처하고 네트워크 내부 구조가 탐지 정확도에 어떤 영향을 미치는지에 대한 연구는 여전히 부족한 상태이다. 따라서 본 논문의 목표는 LSTM의 출력을 조정하여 정보 손실을 줄이고 시간적 의존성을 강화한 Adjusted-LSTM GAN (ALGAN) 모델을 제안함으로써 단변량 및 다변량 시계열 데이터 모두에서 이상치 탐지 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 LSTM의 고질적인 문제인 정보 손실과 기울기 소실(Vanishing Gradient) 문제를 해결하기 위해 **Adjusted-LSTM (ALstm)** 구조를 설계하고 이를 GAN 프레임워크에 통합한 점이다.

ALstm의 중심 아이디어는 LSTM의 출력단에 **Self-Attention** 메커니즘을 적용하여, 입력 시퀀스와 은닉 상태(Hidden States) 중에서 모델 정확도 향상에 가장 중요한 부분에 집중하도록 조정하는 것이다. 구체적으로는 LSTM이 생성한 은닉 상태들 사이의 관계를 분석하여 조정이 필요한 지점을 식별하고, 동시에 입력 시퀀스에서 가장 관련성이 높은 부분을 찾아 이를 은닉 상태에 더해줌으로써 긴 시퀀스에서도 중요한 정보가 유지되도록 설계하였다. 이러한 구조를 Generator와 Discriminator 모두에 적용함으로써, 정상 데이터의 분포를 더욱 정교하게 학습하고 결과적으로 더 정확한 이상치 점수(Anomaly Score)를 산출할 수 있게 하였다.

## 📎 Related Works

논문에서는 기존의 이상치 탐지 방법론을 크게 세 가지 범주로 나누어 설명하며 각각의 한계를 지적한다.

첫째, PCA, PLS와 같은 선형 모델 기반 방법은 데이터가 가우시안 분포를 따르거나 높은 상관관계를 가져야 한다는 제약이 있어 복잡한 실제 시나리오에 부적합하다. KNN과 같은 거리 기반 방법은 이상치의 지속 시간이나 개수에 대한 사전 지식이 필요하며, ABOD나 FB 같은 밀도 추정 기반 모델은 시간적 상관관계를 무시한다는 단점이 있다.

둘째, Auto-Encoder, Encoder-Decoder, LSTM과 같은 딥러닝 기반 모델들은 높은 성능을 보이지만, 단순 LSTM 모델은 긴 시퀀스에서 기울기 소실 문제로 인해 장기 의존성을 학습하는 데 어려움을 겪는다.

셋째, GAN 기반 방법론인 AnoGan은 이미지 도메인에서 성공적이었으나 시간적 의존성을 처리하는 메커니즘이 부족하며, TadGAN이나 HTA-GAN은 LSTM이나 BiLSTM을 사용하지만 여전히 단순한 LSTM 유닛의 한계를 그대로 가지고 있다. 또한 최근의 Transformer 기반 모델(TGAN-AD, Anomaly Transformer)들은 Self-Attention을 사용하지만, 시계열 데이터의 순서와 연속성을 효과적으로 보존하지 못하는 Permutation-invariant 특성이 있어 LSTM보다 시간적 역동성 모델링에 불리할 수 있음을 지적한다.

## 🛠️ Methodology

### 1. Adjusted-LSTM (ALstm) 구조

ALstm은 일반적인 LSTM의 출력 단계에서 발생하는 정보 손실을 방지하기 위해 두 단계의 Attention 레이어를 추가한 구조이다.

* **첫 번째 Attention 레이어 ($A_h$):** LSTM이 각 타임 스텝에서 생성한 은닉 상태 $h_t$들을 Query, Key, Value로 사용하여, 어떤 은닉 상태가 조정되어야 하는지를 식별한다. 계산식은 다음과 같다.
    $$\text{attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **두 번째 Attention 레이어 ($A_x$):** 입력 시퀀스 $x$에서 은닉 상태를 조정하는 데 가장 적합한 부분을 선택적으로 가중치 부여하여 추출한다.
* **최종 조정:** 추출된 $A_x$를 선형 레이어와 $\tanh$ 활성화 함수를 통해 차원을 맞춘 후, $A_h$와 요소별 덧셈(Pointwise Addition)을 수행하여 최종 Adjusted Output을 생성한다.

### 2. ALGAN 전체 파이프라인

ALGAN은 크게 두 가지 하위 프로세스로 구성된다.

**가. 정상 데이터 분포 학습 (Adversarial Training)**
Generator ($G$)와 Discriminator ($D$) 모두 ALstm을 기반으로 구축된다. $G$는 잠재 공간(Latent Space) $Z$에서 샘플링된 노이즈 벡터 $z$를 입력받아 실제와 유사한 시계열 시퀀스를 생성하고, $D$는 입력된 데이터가 실제 데이터인지 $G$가 만든 가짜 데이터인지를 판별한다. 두 네트워크는 다음과 같은 Minimax 게임을 통해 학습된다.
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

**나. 잠재 공간 매핑 및 이상치 탐지**
학습된 $G$는 $Z \to X$ 방향의 매핑만 가능하므로, 실제 데이터 $x$를 다시 $z$로 매핑하기 위한 반복적인 최적화 과정을 거친다. 최적의 $z_\Lambda$를 찾기 위해 다음과 같은 손실 함수 $L$을 최소화하는 방향으로 $z$를 업데이트한다.
$$L(z_\lambda) = (1-\gamma) \cdot L^R(z_\lambda) + \gamma \cdot L^D(z_\lambda)$$
여기서 $L^R$은 실제 값과 생성 값 사이의 포인트별 차이를 계산하는 Residual Loss이며, $L^D$는 Discriminator의 중간 레이어 특징 값을 비교하는 Discrimination Loss이다.

최종적으로 이상치 점수 $A(x)$는 최적화된 $z_\Lambda$를 통해 얻은 잔차 점수 $R(x)$와 판별 점수 $D(x)$의 가중 합으로 결정된다.
$$A(x) = (1-\gamma) \cdot R(x) + \gamma \cdot D(x)$$

## 📊 Results

### 실험 설정

* **데이터셋:** 46개의 단변량 시계열 데이터셋(NAB collection)과 1개의 대규모 다변량 데이터셋(SWaT system, 약 946K 데이터 포인트)을 사용하였다.
* **비교 모델:** VanLstm, GMM, OcSVM, AutoEncoder, IsoForest, MadGan, BiLstmGan, CnnGan 등 8가지 모델과 비교하였다.
* **평가 지표:** Accuracy (Ac), Precision (Pr), Recall (Re), F1-Score, Cohen Kappa (CK), AUC를 측정하였다.

### 주요 결과

1. **NAB 데이터셋 (단변량):** ALGAN은 모든 지표에서 baseline 모델들을 압도하였다. 특히 두 번째로 성능이 좋았던 AutoEncoder와 비교했을 때 Precision에서 7.12%, F1-score에서 6.461%, CK에서 8.224%의 유의미한 향상을 보였다.
2. **SWaT 데이터셋 (다변량):** 다변량 데이터에서도 ALGAN은 Recall, F1, CK, AUC 지표에서 가장 높은 성능을 기록하였다. AutoEncoder가 Precision 면에서는 약간 앞섰으나, ALGAN이 훨씬 높은 Recall을 기록하며 전반적으로 균형 잡힌 F1-score를 달성하였다.
3. **데이터 크기 및 도메인 분석:** 분석 결과 ALGAN은 데이터셋의 크기가 작을수록(5000개 미만) 타 모델 대비 우위가 뚜렷했으며, 데이터의 도메인에 관계없이 일관되게 높은 성능을 보였다. 이는 ALstm 구조가 적은 데이터 환경에서도 정보 손실을 효과적으로 억제함을 시사한다.

## 🧠 Insights & Discussion

본 연구의 결과는 GAN의 생성 모델이 데이터의 일반적인 분포를 학습하는 능력이 시계열 이상치 탐지에 매우 유용함을 입증한다. 특히 ALstm의 도입은 단순 LSTM이 겪는 '망각 문제(Forgetting problem)'를 해결하여 긴 시퀀스에서도 시간적 의존성을 더 정확하게 복원할 수 있게 하였다.

기존 GAN 기반 모델(MadGan, BiLstmGan 등)보다 성능이 뛰어난 이유는, 단순히 네트워크를 깊게 쌓는 것이 아니라 Attention 메커니즘을 통해 입력값과 은닉 상태를 능동적으로 조정함으로써 정보 손실을 최소화했기 때문이다. 또한, AutoEncoder 계열 모델들이 압축 과정에서 발생하는 정보 손실(Lossy compression)로 인해 정밀도가 떨어지는 반면, ALGAN은 적대적 학습을 통해 더 충실한 데이터 표현(Faithful representation)을 학습함으로써 이를 극복하였다.

다만, 논문에서도 언급되었듯이 GAN 기반 모델의 공통적인 한계인 최적의 윈도우 길이(Window length) 설정 문제와 학습 과정의 불안정성(Instability)은 여전히 해결해야 할 과제로 남아 있으며, 학습 에폭(Epoch) 수에 따른 민감도에 대한 추가적인 연구가 필요하다.

## 📌 TL;DR

본 논문은 LSTM의 출력단에 Self-Attention을 적용하여 정보 손실을 줄인 **Adjusted-LSTM (ALstm)**을 제안하고, 이를 GAN 구조에 통합한 **ALGAN** 모델을 통해 시계열 이상치 탐지 성능을 극대화하였다. 47개의 다양한 실세계 데이터셋 실험 결과, 단변량과 다변량 모두에서 기존의 LSTM, AutoEncoder, 타 GAN 기반 모델들보다 우수한 성능을 보였으며, 특히 데이터 크기가 작은 환경에서 탁월한 효율성을 입증하였다. 이 연구는 향후 정밀한 모니터링이 필요한 금융, 헬스케어, 사이버 보안 분야의 이상 탐지 시스템에 적용될 가능성이 높다.
