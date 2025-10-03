# Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective

Zexin Wang, Changhua Pei, Minghua Ma, Xin Wang, Zhihan Li, Dan Pei, Saravan Rajmohan, Dongmei Zhang, Qingwei Lin, Haiming Zhang, Jianhui Li, Gaogang Xie

## 🧩 Problem to Solve

시계열 이상 탐지(Anomaly Detection, AD)는 웹 시스템 모니터링에 필수적입니다. 기존 VAE(Variational Autoencoder) 기반 방법론은 우수한 노이즈 제거 능력으로 AD 분야에서 인기를 얻었으나, 다음과 같은 주요 문제에 직면합니다:

- **장주기 이질적 패턴 및 단주기 세부 추세 동시 포착의 어려움**: 기존 VAE 기반 방법은 다양한 주기 내에서 모양이 변화하는 이질적 패턴(예: Figure 1의 파란색 타원)과 국부적인 세부 패턴(예: Figure 1의 파란색 사각형)을 동시에 효과적으로 포착하지 못합니다.
- **주파수 영역 정보의 누락**: 이러한 실패의 근본 원인은 정상 데이터의 재구성에 필수적인 일부 주파수 영역 정보가 기존 모델에서 누락되기 때문입니다 (Figure 2(b)).
- **노이즈와 사용의 어려움**: 조건부 VAE(CVAE)의 조건으로 전체 윈도우의 주파수 정보를 직접 활용할 경우, 너무 많은 하위 주파수가 노이즈를 유발하여 효과적인 재구성을 방해합니다.

## ✨ Key Contributions

- 기존 VAE 기반 이상 탐지 모델이 이질적 주기 패턴과 세부적인 추세 패턴을 동시에 포착하지 못하는 문제를 분석하고, 그 원인이 주파수 영역 정보의 누락에 있음을 규명했습니다.
- 주파수 정보를 활용하여 VAE 모델을 체계적으로 개선함으로써, VAE 기반 접근 방식이 시계열 이상 탐지 분야에서 다시금 최첨단 성능을 달성하도록 했습니다. 이는 VAE 기반 방법이 이상 데이터와 정상 데이터가 혼합된 훈련 데이터를 본질적으로 처리할 수 있다는 점에서 중요합니다.
- 전역 및 지역 주파수 정보를 혁신적으로 통합한 새로운 비지도 시계열 이상 탐지 방법인 **FCVAE (Frequency-enhanced Conditional Variational AutoEncoder)**를 제안했습니다.
- 공개 데이터셋과 대규모 클라우드 시스템에서의 평가를 통해 FCVAE가 최신 방법론들을 F1 점수 기준으로 최대 40%(공개 데이터셋), 10%(실제 웹 시스템) 이상 크게 능가함을 입증하여 실제 적용 가능성을 확인했습니다.
- 포괄적인 제거 연구(ablation studies)를 통해 모델의 우수한 성능에 대한 심층적인 분석과 이유를 제시했습니다.

## 📎 Related Works

- **전통적인 통계 방법**: SPOT, STL, FFT 등 시계열 데이터 처리에 강점이 있는 방법들.
- **지도 학습 방법**: Opprentice, SRCNN, TFAD 등 고품질 라벨을 기반으로 이상 특징을 학습하고 분류기로 식별하는 방법들.
- **비지도 학습 방법**:
  - **재구성 기반 방법**: DONUT, Buzz, ACVAE, VQRAE, AnoTransfer 등 저차원 표현을 학습하고 데이터의 "정상 패턴"을 재구성하여 재구성 오차로 이상을 탐지합니다.
  - **예측 기반 방법**: Informer, Anomaly-Transformer 등 과거 데이터를 기반으로 지표의 정상 값을 예측하고 예측 오차로 이상을 탐지합니다.
- **트랜스포머 기반 방법**: Anomaly-Transformer, TimesNet, Fedformer 등 최근 주목받는 트랜스포머 아키텍처를 활용한 방법들.
- **전이 학습 방법**: 시계열 이상 탐지에서 전이 학습을 활용하는 연구들.

## 🛠️ Methodology

FCVAE는 데이터 전처리, 훈련, 테스트의 세 가지 주요 구성 요소로 이루어져 있으며, CVAE 프레임워크에 전역 및 지역 주파수 정보를 조건으로 통합합니다.

1. **데이터 전처리 (Data Preprocessing)**

   - **표준화 및 결측/이상점 채우기**: 기존 연구에서 효과가 입증된 기술을 적용합니다.
   - **데이터 증강 (Data Augmentation)**: 주로 이상 데이터에 초점을 맞춥니다.
     - **패턴 변이 (Pattern Mutation)**: 다른 곡선에서 두 윈도우를 결합하여 접합부를 이상점으로 만듭니다.
     - **값 변이 (Value Mutation)**: 윈도우 내 일부 지점을 무작위로 비정상적인 값으로 변경합니다. 이를 통해 모델이 이상 데이터를 처리하는 방법을 학습하도록 돕습니다.

2. **네트워크 아키텍처 (Network Architecture)** (Figure 4)

   - **인코더(Encoder) 및 디코더(Decoder)**: 시계열 $x$를 잠재 공간 $z$로 인코딩하고 다시 재구성합니다.
   - **조건 추출 블록 (Condition Extraction Block)**:
     - **전역 주파수 정보 추출 모듈 (GFM: Global Frequency information extraction Module)** (Figure 7):
       - 전체 윈도우 $x$에 대해 FFT (Fast Fourier Transform) 변환 $F(x)$을 적용하여 전역 주파수 정보를 추출합니다.
       - 선형 레이어(Dense)를 사용하여 유용한 주파수 정보를 필터링하고, 드롭아웃(Dropout) 레이어를 통해 누락된 주파수 정보 학습 능력을 향상시킵니다.
       - 출력: $f_{\text{global}} = \text{Dropout}(\text{Dense}(F(x)))$
     - **지역 주파수 정보 추출 모듈 (LFM: Local Frequency information extraction Module)** (Figure 5):
       - GFM이 마지막 지점에 대한 충분한 주의를 기울이지 못하는 문제를 해결하기 위해 도입됩니다.
       - 전체 윈도우 $x$를 슬라이딩하여 여러 작은 윈도우 $x_{\text{sw}}$를 얻습니다.
       - 각 작은 윈도우에 FFT 및 주파수 정보 추출을 적용합니다.
       - 가장 최근의 작은 윈도우는 쿼리 $Q$로 사용되고, 나머지 작은 윈도우는 키 $K$와 값 $V$로 사용됩니다.
       - **타겟 어텐션 (Target Attention)** 메커니즘을 통해 각 작은 윈도우의 주파수 정보에 가중치를 부여하여 가장 중요한 지역 주파수 정보를 추출합니다.
       - 선형 레이어 및 드롭아웃이 적용됩니다.
       - 출력: $f_{\text{local}} = \text{Dropout}(\text{FeedFawrd}((\sigma(Q \cdot K^\top) \cdot V)))$, 여기서 $x_{\text{sw}} \in R^{n \times k}$ (크기 $k$의 작은 윈도우 $n$개), $Q = \text{Select}(\text{Dense}(F(x_{\text{sw}})))$, $K,V = \text{Dense}(F(x_{\text{sw}}))$ 입니다.
   - **FCVAE 작동**: $\mu, \sigma = \text{Encoder}(x, \text{LFM}(x), \text{GFM}(x))$, $z = \text{Sample}(\mu, \sigma)$, $\mu_x, \sigma_x = \text{Decoder}(z, \text{LFM}(x), \text{GFM}(x))$

3. **훈련 및 테스트 (Training and Testing)**
   - **CM-ELBO (CVAE-based Modified Evidence Lower Bound)**: VAE의 M-ELBO를 CVAE에 적용하여 이상/결측 데이터의 영향을 완화합니다.
     $$ L=E*{q*\phi(z|x,c)} \left[ \sum_{w=1}^W \alpha_w \log p_\theta(x_w|z,c) + \beta \log p_\theta(z) - \log q_\phi(z|x,c) \right] $$
   - **결측 데이터 주입 (Missing Data Injection)**: VAE에서 일반적으로 사용되는 기술을 적용합니다.
   - **마지막 지점 마스킹 (Masking the last point)**: 주파수 조건 추출 시 윈도우의 마지막 지점을 0으로 마스킹하여, 마지막 지점의 이상이 전체 주파수 정보에 미치는 영향을 완화합니다.
   - **테스트**: MCMC(Markov Chain Monte Carlo) 기반 결측치 대체 알고리즘을 사용하여 마지막 지점의 정상 값을 얻고, 재구성 확률을 이상 점수(Anomaly Score)로 사용합니다.
     $$ \text{AnomalyScore} = -E*{q*\phi(z|x,c)}[\log p_\theta(x|z,c)] $$

## 📊 Results

- **전반적인 성능**: FCVAE는 Yahoo, KPI, WSD, NAB 네 가지 데이터셋 모두에서 모든 기준선(baseline) 방법론을 능가했습니다. `best F1` 점수에서 6.45%에서 14.14%, `delay F1` 점수에서 4.98%에서 38.68%까지 향상된 성능을 보였습니다.
- **CVAE 조건 유형 비교**: 주파수 정보를 조건으로 사용하는 것이 타임스탬프 또는 시간 도메인 정보를 사용하는 것보다 우수한 성능을 나타냈습니다 (Figure 9(a)). 주파수 정보가 가치 있는 보완적 사전 정보 역할을 합니다.
- **FVAE 및 FCVAE 비교**: 주파수 정보를 VAE 입력으로 사용하는 FVAE보다 FCVAE(주파수 정보를 CVAE의 조건으로 사용)가 더 나은 성능을 보였습니다. 이는 CVAE 아키텍처가 조건 정보를 더 효과적으로 활용함을 시사합니다 (Figure 9(b)).
- **GFM 및 LFM의 효과**: GFM과 LFM 단독 사용 시 VAE보다 성능이 향상되며, 두 모듈을 함께 사용했을 때 시너지 효과로 인해 성능이 더욱 우수해짐을 확인했습니다 (Figure 9(c)). 이는 전역 및 지역 주파수 정보 모두 이상 탐지에 중요함을 의미합니다.
- **어텐션 메커니즘의 역할**: LFM의 어텐션 메커니즘은 각 작은 윈도우에 적절한 가중치를 할당하여 성능 향상에 필수적임을 입증했습니다. 어텐션이 없을 경우 성능이 저하됩니다 (Figure 9(d), Figure 10).
- **프레임워크 핵심 기술**: CM-ELBO, 마지막 지점 마스킹, 데이터 증강 등 제안된 핵심 기술들이 모두 성능 향상에 기여하며, 특히 CM-ELBO가 가장 중요한 역할을 했습니다 (Table 2).
- **매개변수 민감도**: FCVAE 모델은 윈도우 크기, 임베딩 차원, 결측 데이터 주입 비율, 데이터 증강 비율 등 다양한 매개변수 설정에서 안정적이고 우수한 결과를 달성했습니다 (Figure 11).
- **실제 시스템 도입 효과**: 대규모 클라우드 시스템에 FCVAE를 도입한 결과, 기존 탐지기 대비 F1 점수에서 10.9%, 지연 F1 점수에서 11.1%의 상당한 성능 향상을 보였습니다. 또한, 초당 1000개 이상의 데이터 포인트를 처리하는 높은 효율성을 입증했습니다 (1195.7 points/second).

## 🧠 Insights & Discussion

- 기존 VAE는 주파수 영역 정보를 간과하여 복잡한 시계열 패턴(이질적 주기성, 세부 추세)을 포착하는 데 한계가 있었습니다.
- 주파수 정보는 VAE 기반 이상 탐지에서 가치 있고 상보적인 사전 정보(prior)로 작용하여 재구성 정확도를 크게 높입니다.
- CVAE 아키텍처는 조건부 정보를 VAE 입력으로 직접 사용하는 것보다 훨씬 효과적으로 통합합니다.
- 전역 주파수 정보(장기 패턴)와 지역 주파수 정보(단기, 세부 패턴)를 동시에 활용하는 것이 이상 탐지 성능에 결정적이며, 이 둘의 시너지 효과가 매우 강력합니다.
- LFM의 타겟 어텐션 메커니즘은 최신 시점의 세부 패턴에 집중하여 가장 유용한 지역 주파수 정보를 선별적으로 추출하는 데 중요한 역할을 합니다.
- 훈련 시 CM-ELBO, 마지막 지점 마스킹, 비정상 데이터 증강과 같은 기술들은 비지도 학습 환경에서 모델의 견고성과 이상 처리 능력을 향상시키는 데 필수적입니다.
- FCVAE는 다양한 매개변수 설정에 대해 견고한 성능을 보이며, 실제 대규모 시스템에 적용될 만큼 효율적이고 실용적입니다.

## 📌 TL;DR

- **문제**: 기존 VAE 기반 시계열 이상 탐지 모델은 주파수 정보 누락으로 인해 복잡하고 이질적인 주기 패턴 및 세부적인 추세를 효과적으로 포착하지 못했습니다.
- **해결책**: FCVAE(Frequency-enhanced Conditional Variational AutoEncoder)는 CVAE 프레임워크를 기반으로 전역 주파수 정보(GFM)와 지역 주파수 정보(LFM)를 조건으로 활용합니다. LFM은 타겟 어텐션 메커니즘을 통해 최신 시점의 세부 추세에 집중합니다. 훈련 시 CM-ELBO, 이상 데이터 증강, 마지막 지점 마스킹과 같은 기술을 적용하여 모델의 견고성을 높였습니다.
- **주요 결과**: FCVAE는 공개 데이터셋과 실제 클라우드 시스템 모두에서 최신 이상 탐지 방법들보다 F1 점수 기준 최대 14% 향상된 성능을 보였으며, 높은 효율성과 매개변수 견고성을 입증했습니다. 주파수 정보의 조건부 활용과 CVAE의 아키텍처가 복잡한 시계열 패턴을 효과적으로 학습하는 핵심 요소임이 밝혀졌습니다.
