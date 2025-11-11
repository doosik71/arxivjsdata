# ANOMALYTRANSFORMER: TIMESERIESANOMALY DETECTION WITHASSOCIATIONDISCREPANCY

Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long

## 🧩 Problem to Solve

시계열 데이터에서 이상 지점을 비지도 학습 방식으로 탐지하는 것은 어려운 문제입니다. 기존의 방법들은 주로 점별(pointwise) 표현 학습이나 쌍별(pairwise) 연관성 모델링에 초점을 맞췄지만, 복잡한 시계열 동역학을 포괄적으로 이해하기에는 불충분합니다. 특히 이상치는 희귀하여 모델이 정보력이 풍부한 표현을 학습하고, 정상과 이상 지점을 구별할 수 있는 효과적인 기준을 도출하기 어렵습니다.

## ✨ Key Contributions

- **연관성 불일치(Association Discrepancy) 개념 제안:** 이상치의 희소성으로 인해 이상 지점은 전체 시계열과의 의미 있는 연관성을 형성하기 어렵고, 주로 인접한 시점과 연관성이 집중된다는 핵심 관찰에 기반하여, 사전 연관성(prior-association)과 시계열 연관성(series-association) 간의 연관성 불일치를 새로운 이상 탐지 기준으로 제안합니다.
- **Anomaly Transformer 및 Anomaly-Attention 메커니즘 개발:** 연관성 불일치를 효과적으로 모델링하기 위해, 사전 연관성(학습 가능한 가우시안 커널 기반)과 시계열 연관성(셀프 어텐션 기반)을 동시에 학습하는 2-브랜치 구조의 Anomaly-Attention 메커니즘을 갖춘 Anomaly Transformer를 제안합니다.
- **미니맥스(Minimax) 학습 전략 도입:** 연관성 불일치의 정상-이상 구별력을 극대화하기 위해 미니맥스 전략을 고안했습니다. 이는 사전 연관성이 시계열 연관성에 근접하도록 유도하면서, 동시에 시계열 연관성이 연관성 불일치를 최대화하도록 학습합니다.
- **최첨단(State-of-the-Art) 성능 달성:** 서비스 모니터링, 우주 탐사, 수처리 등 세 가지 응용 분야의 6개 시계열 이상 탐지 벤치마크에서 기존 모델들을 능가하는 최첨단 성능을 달성했습니다.

## 📎 Related Works

- **밀도 기반(Density-estimation) 방법:** LOF (Breunig et al., 2000), COF (Tang et al., 2002), DAGMM (Zong et al., 2018), MPPCACD (Yairi et al., 2017).
- **클러스터링 기반(Clustering-based) 방법:** OC-SVM (Schölkopf et al., 2001), SVDD (Tax & Duin, 2004), Deep SVDD (Ruff et al., 2018), THOC (Shen et al., 2020), ITAD (Shin et al., 2020).
- **재구성 기반(Reconstruction-based) 모델:** LSTM-VAE (Park et al., 2018), OmniAnomaly (Su et al., 2019), InterFusion (Li et al., 2021), GAN 기반 (BeatGAN; Zhou et al., 2019).
- **자기회귀 기반(Autoregression-based) 모델:** VAR (Anderson & Kendall, 1976), LSTM (Hundman et al., 2018), CL-MPPCA (Tariq et al., 2019).
- **그래프 기반(Graph-based) 모델:** Cheng et al. (2008, 2009), GNN (Zhao et al., 2020; Deng & Hooi, 2021).
- **부분 시계열 기반(Subsequence-based) 모델:** Boniol & Palpanas (2020).
- **시계열 분석을 위한 Transformer:** Informer (Zhou et al., 2021), Autoformer (Wu et al., 2021), GTA (Chen et al., 2021).

## 🛠️ Methodology

Anomaly Transformer는 Transformer 아키텍처를 시계열 이상 탐지에 맞게 재설계합니다.

1. **전반적인 아키텍처:** Anomaly-Attention 블록과 피드포워드 레이어를 교대로 쌓아 심층적인 다단계 특징에서 기본 연관성을 학습합니다. $l$번째 레이어의 출력 $X^l \in \mathbb{R}^{N \times d_{model}}$은 다음과 같이 정의됩니다:
   $$Z^l = \text{Layer-Norm}(\text{Anomaly-Attention}(X^{l-1}) + X^{l-1})$$
   $$X^l = \text{Layer-Norm}(\text{Feed-Forward}(Z^l) + Z^l)$$
2. **Anomaly-Attention 메커니즘:**
   - **사전 연관성 ($P^l$):** 학습 가능한 스케일 $\sigma_i$를 가진 가우시안 커널 $G(|j-i|; \sigma_i) = \frac{1}{\sqrt{2\pi\sigma_i}} \exp(-\frac{|j-i|^2}{2\sigma_i^2})$을 사용하여 각 시점이 인접한 시점에 더 집중하는 편향을 모델링합니다.
   - **시계열 연관성 ($S^l$):** 원본 시계열에서 적응적으로 연관성을 학습하는 표준 셀프 어텐션 메커니즘 (Softmax($QK^T / \sqrt{d_{model}}$))을 사용합니다.
   - 이 두 연관성은 각 시점의 시간적 의존성을 반영하며, 그 차이($P^l$와 $S^l$)가 정상-이상 구별력을 가집니다.
3. **연관성 불일치(Association Discrepancy) 정의:**
   - 사전 연관성 $P$와 시계열 연관성 $S$ 간의 대칭적인 쿨백-라이블러(KL) 발산으로 정의됩니다.
   - 여러 레이어의 연관성 불일치를 평균하여 최종적인 점별 연관성 불일치 $\text{AssDis}(P,S;X) \in \mathbb{R}^{N \times 1}$를 계산합니다:
     $$\text{AssDis}(P,S;X) = \left[ \frac{1}{L} \sum_{l=1}^{L} (\text{KL}(P^l_{i,:} \Vert S^l_{i,:}) + \text{KL}(S^l_{i,:} \Vert P^l_{i,:})) \right]_{i=1, \dots, N}$$
4. **미니맥스 연관성 학습 전략:**
   - 모델은 재구성 손실과 연관성 불일치 손실을 포함하는 총 손실 함수 $L_{Total}(\hat{X}, P, S, \lambda; X) = \|\hat{X} - X\|^2_F - \lambda \times \|\text{AssDis}(P, S; X)\|_1$를 사용하여 최적화됩니다.
   - **최소화 단계(Minimize Phase):** 사전 연관성 $P$가 시계열 연관성 $S$ (그래디언트 역전파를 중단한 $S_{\text{detach}}$)에 근접하도록 하여, $P$가 다양한 시간 패턴에 적응하게 합니다. ($L_{Total}(\hat{X}, P, S_{\text{detach}}, -\lambda; X)$)
   - **최대화 단계(Maximize Phase):** 시계열 연관성 $S$가 연관성 불일치를 확대하도록 최적화됩니다 (사전 연관성 $P$는 $P_{\text{detach}}$로 고정). 이는 $S$가 비인접 영역에 더 주목하도록 강제하여, 이상치 재구성을 어렵게 만들고 정상-이상 구별력을 증폭시킵니다. ($L_{Total}(\hat{X}, P_{\text{detach}}, S, \lambda; X)$)
5. **연관성 기반 이상 기준(Anomaly Criterion):**
   - 재구성 오차와 정규화된 연관성 불일치를 결합하여 최종 이상 점수를 계산합니다:
     $$\text{AnomalyScore}(X) = \text{Softmax}(-\text{AssDis}(P,S;X)) \odot \left[\|X_{i,:} - \hat{X}_{i,:}\|_2^2\right]_{i=1, \dots, N}$$
   - 여기서 $\odot$는 요소별 곱셈입니다. 재구성 성능이 좋을수록 이상 연관성 불일치가 감소하는 경향이 있어, 더 높은 이상 점수를 도출합니다.

## 📊 Results

- **최첨단 성능:** SMD, PSM, MSL, SMAP, SWaT, NeurIPS-TS의 6개 벤치마크 데이터셋에서 일관되게 최첨단 F1-score와 AUC 값을 달성했습니다.
- **NeurIPS-TS 벤치마크:** 점별 이상치와 패턴별 이상치를 모두 포함하는 다양한 이상치 유형에 대해서도 강력한 성능을 보여, 모델의 범용성을 입증했습니다.
- **어블레이션 연구:**
  - 제안된 연관성 기반 기준은 순수 재구성 기준에 비해 평균 F1-score를 18.76% 향상시켰습니다.
  - 학습 가능한 사전 연관성($\sigma$)과 미니맥스 전략은 각각 평균 F1-score를 8.43% 및 7.48% 추가적으로 향상시켜, 각 구성 요소의 효과를 검증했습니다.
- **모델 분석:**
  - **이상 기준 시각화:** 제안된 연관성 기반 기준이 기존 재구성 기준보다 정상 및 이상 지점을 더 명확하게 구별하며, 오탐율을 줄이는 데 기여함을 시각적으로 확인했습니다.
  - **사전 연관성 시각화:** 이상치의 경우 학습된 $\sigma$ 값이 더 작게 나타나, 이상치 연관성이 인접 지점에 집중된다는 가설을 뒷받침합니다.
  - **최적화 전략 분석:** 미니맥스 전략이 직접 최대화 방식보다 정상-이상 지점 간의 인접 연관성 가중치 차이(contrast value)를 더 크게 증폭시킴을 통계적으로 보여주었습니다.

## 🧠 Insights & Discussion

Anomaly Transformer는 Transformer의 강력한 전역 표현 및 장거리 관계 모델링 능력을 활용하여 시계열 이상 탐지의 새로운 가능성을 제시합니다. 특히, 이상치의 희소성이라는 본질적인 특성을 '연관성 불일치'라는 개념으로 전환하여, 정상-이상 구별력을 내재적으로 높이는 독창적인 접근 방식을 제안합니다. 미니맥스 학습 전략은 이러한 구별력을 더욱 증폭시키며, 재구성 손실과 연관성 불일치 기준을 효과적으로 결합함으로써, 기존의 점별 재구성 오차에만 의존하던 한계를 극복합니다.

모델은 다양한 유형의 이상치와 실제 애플리케이션 데이터셋에서 탁월한 성능을 보였으며, 이는 제안된 연관성 기반 기준의 강점을 명확히 보여줍니다. Anomaly-Attention 메커니즘과 학습 가능한 사전 연관성은 모델의 적응성과 해석 가능성을 높여줍니다.

**한계점:**

- Transformer의 창 크기에 대한 2차 복잡도는 대규모 시계열 처리 시 메모리 및 계산 효율성 측면에서 여전히 트레이드오프가 필요합니다.
- 제안된 Anomaly Transformer에 대한 고전적인 자기회귀 및 상태 공간 모델 분석에 비추어 볼 때, 더 심도 깊은 이론적 연구가 필요합니다.

## 📌 TL;DR

본 논문은 시계열 이상 탐지 문제에 Transformer를 도입하여 **연관성 불일치(Association Discrepancy)**라는 새로운 이상 탐지 기준을 제안합니다. 이상치 연관성은 인접 지점에 집중되고(사전 연관성), 정상치 연관성은 더 광범위하다는 핵심 관찰을 기반으로, 이를 **Anomaly-Attention** 메커니즘을 통해 모델링합니다. 또한, **미니맥스 학습 전략**을 사용하여 정상과 이상 지점 간의 연관성 구별력을 극대화하며, 재구성 오차와 연관성 불일치를 결합한 최종 이상 점수를 계산합니다. 이 **Anomaly Transformer**는 6개 벤치마크 데이터셋에서 최첨단 성능을 달성하며, 시계열 이상 탐지에 대한 효과적이고 해석 가능한 접근 방식을 제공합니다.
