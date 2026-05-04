# ON THE EFFICIENCY OF INTEGRATING SELF-SUPERVISED LEARNING AND META-LEARNING FOR USER-DEFINED FEW-SHOT KEYWORD SPOTTING

Wei-Tsung Kao, Yuan-Kuei Wu, Chia-Ping Chen, Zhi-Sheng Chen, Yu-Pao Tsai, Hung-Yi Lee (2022)

## 🧩 Problem to Solve

본 논문은 사용자가 직접 정의한 키워드를 감지하는 **User-defined Keyword Spotting (KWS)** 문제를 해결하고자 한다. 일반적인 KWS 시스템은 제조사가 미리 정의한 키워드에 대해 대규모 데이터를 수집하여 학습시키지만, 이는 사용자 개인화라는 측면에서 한계가 있다.

사용자가 정의하는 키워드의 경우, 사용자가 많은 양의 음성 샘플을 제공하는 것이 현실적으로 불가능하므로 이는 전형적인 **Few-shot Learning** 문제로 귀결된다. 기존 연구들은 대규모 라벨링 데이터를 이용한 전이 학습이나 Self-supervised Learning (SSL) 모델을 활용하려 했으나, SSL과 Meta-learning의 결합이 상호 보완적인지, 그리고 어떤 조합이 Few-shot KWS에 가장 효과적인지에 대해서는 명확히 밝혀지지 않았다. 따라서 본 연구의 목표는 다양한 SSL 모델과 Meta-learning 알고리즘을 체계적으로 조합하여 그 효율성을 분석하고 최적의 조합을 찾는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1.  **체계적인 조합 분석**: 다양한 최신 SSL 모델(CPC, TERA, Wav2Vec2, HuBERT, WavLM)과 전형적인 Meta-learning 알고리즘(Optimization-based 및 Metric-based)을 조합하여 Few-shot KWS 성능을 실험적으로 검증하였다.
2.  **최적 조합 발견**: 실험 결과, **HuBERT** SSL 모델과 **Matching network** 알고리즘의 조합이 1-shot 및 5-shot 시나리오 모두에서 최고의 성능을 보이며, 적은 수의 예제 변화에도 강건(robust)함을 입증하였다.
3.  **상호 보완성 입증**: SSL을 통한 사전 학습과 Meta-learning의 효과가 가산적(additive)임을 보였으며, 이를 통해 새로운 키워드에 대해 더욱 변별력 있는 임베딩을 생성할 수 있음을 확인하였다.

## 📎 Related Works

기존의 Few-shot KWS 접근 방식은 크게 두 가지 범주로 나뉜다.

-   **라벨링된 데이터로부터의 전이 학습**: MAML이나 Prototypical network를 사용하여 인코더를 학습시키거나, LibriSpeech와 같은 대규모 데이터셋에서 soft-triple loss 등을 이용해 임베딩 모델을 학습시키는 방식이다. 그러나 이러한 방식은 오디오, 전사 데이터, forced aligner 등의 준비 비용이 높고, TTS(Text-to-Speech)로 합성한 데이터를 사용할 경우 도메인 불일치(domain mismatch) 문제가 발생할 수 있다.
-   **라벨링되지 않은 데이터 활용 (SSL)**: Wav2Vec 2.0과 같은 SSL 모델을 특징 추출기로 사용하는 방식이다. 대규모 무라벨 데이터에서 유용한 표현을 배울 수 있다는 장점이 있으나, 기존 연구에서는 전체 데이터셋을 사용할 때의 성능에 집중하여 1-shot과 같은 극소량의 데이터 환경에서의 성능은 낮게 나타났다.

본 논문은 이러한 두 접근법이 서로 보완적인지, 그리고 특정 SSL 모델이나 알고리즘 선택에 따라 그 효과가 달라지는지를 심층적으로 분석함으로써 기존 연구와 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
KWS 시스템은 인코더 $\theta_e$와 분류기 $\theta_c$로 구성된다. 입력 음성 신호가 인코더를 통과하여 벡터 시퀀스를 생성하고, 이를 분류기가 받아 각 키워드에 대한 사후 확률을 출력한다. 인코더는 사전 학습된 SSL 모델을 사용하거나 랜덤 초기화된 파라미터를 사용할 수 있다.

### Self-supervised Learning (SSL)
본 연구에서는 다음 5가지 SSL 모델을 인코더로 검토하였다.
-   **CPC**: 미래 프레임의 잠재 표현을 예측하는 LSTM 기반 모델이다.
-   **TERA**: 스펙트로그램의 일부를 마스킹하고 이를 복원하는 Transformer 기반 모델이다.
-   **Wav2Vec2**: 입력 표현의 일부를 마스킹하고 CPC와 유사한 손실 함수로 복원하는 모델이다.
-   **HuBERT**: MFCC 프레임을 클러스터링하고, 마스킹된 위치의 클러스터 ID를 예측한다.
-   **WavLM**: HuBERT와 유사하나 노이즈가 섞인 데이터를 포함하여 학습된다.

인코더의 각 레이어에서 출력되는 시퀀스 $\{h_t\}_{t=1}^T$를 시간축으로 평균하여 단일 표현 $\bar{h}$를 얻고, 여러 레이어의 $\bar{h}$를 학습 가능한 가중치로 합산하여 최종 분류기에 입력한다.

### Meta-learning
Meta-learning은 소수의 예제로 새로운 태스크에 빠르게 적응하는 것을 목표로 한다. 본 논문에서는 두 가지 유형의 알고리즘을 사용하였다.

#### 1. Optimization-based methods (MAML, ANIL, BOIL, Reptile)
이 방법들은 빠른 적응을 위한 최적의 초기 파라미터를 찾는 것이 목적이다.
-   **Inner Loop**: 서포트 셋 $\hat{S}_i$를 사용하여 태스크별 모델 $\theta_i$를 업데이트한다.
    $$\theta_i \leftarrow \theta - \alpha \nabla_\theta L_{\hat{S}_i}(f_\theta)$$
    여기서 $L$은 교차 엔트로피 손실 함수(Cross-Entropy Loss)이다.
-   **Outer Loop**: 쿼리 셋 $\hat{Q}_i$에서의 손실을 바탕으로 전체 파라미터 $\theta$를 업데이트한다.
    $$\theta \leftarrow \theta - \beta \sum_i \nabla_\theta L_{\hat{Q}_i}(f_{\theta_i})$$
-   **변형**: MAML은 전체 모델을 업데이트하며, ANIL은 인코더 $\theta_e$를 고정하고 분류기만 업데이트한다. BOIL은 반대로 분류기를 고정하고 인코더를 업데이트한다. Reptile은 그라디언트 계산 대신 $\theta$와 $\theta_i$의 차이를 직접 이용한다.

#### 2. Metric-based methods (Prototypical, Relational, Matching networks)
이 방법들은 동일한 키워드의 임베딩이 서로 가깝게 위치하도록 학습한다.
-   **Prototypical Network**: 각 키워드의 예제 임베딩을 평균 내어 프로토타입(centroid) $h_w$를 생성하고, 쿼리와의 $L_2$ 거리를 측정하여 분류한다.
-   **Relational Network**: 쿼리와 프로토타입을 결합하여 관계 점수(scalar)를 출력하는 별도의 네트워크 $f_{\theta_c}$를 학습시킨다.
-   **Matching Network**: 어텐션 메커니즘을 통해 서포트 셋과 쿼리 간의 가중치 합을 계산하며, $L_2$ 거리를 기반으로 확률을 도출한다.

## 📊 Results

### 실험 설정
-   **데이터셋**: Google Speech Commands V2 및 WHAM! 노이즈 데이터셋을 사용하였다.
-   **태스크**: 12-way K-shot (키워드 10종, Unknown 1종, Silence 1종) 분류 문제로 설정하였다. $K$값은 1과 5로 설정하였다.
-   **평가 지표**: 평균 정확도(Accuracy) 및 표준편차(Standard Deviation)를 측정하였다.

### 주요 결과
-   **알고리즘 비교**: 일반적으로 Metric-based 방법이 Optimization-based 방법보다 우수한 성능을 보였다. 특히 **Matching network**가 1-shot과 5-shot 모두에서 가장 높은 정확도를 기록하였다.
-   **SSL 모델 비교**: **HuBERT**가 1-shot 및 5-shot 환경에서 가장 뛰어난 성능을 보였다. 또한 HuBERT는 인코더를 고정(fix-encoder)했을 때 성능이 크게 향상되는 경향을 보였는데, 이는 HuBERT의 표현력이 Meta-learning에 매우 적합함을 시사한다.
-   **강건성**: Table 2의 표준편차 분석 결과, Matching network가 평균 성능이 가장 높을 뿐만 아니라 지원 세트(support set)의 변화에 가장 강건한 모습을 보였다.
-   **데이터셋 확장성**: Common Voice 데이터셋에서도 동일하게 HuBERT와 Matching network의 조합이 가장 우수한 성능을 기록하여, 제안한 결론이 데이터셋에 관계없이 일반적임을 입증하였다.

## 🧠 Insights & Discussion

### SSL과 Meta-learning의 시너지
본 논문은 SSL과 Meta-learning이 서로 독립적으로 기여하며 그 효과가 가산적(additive)임을 강조한다. 이를 증명하기 위해 SSL 없이 Meta-learning만 수행한 경우(meta only)와 Meta-learning 없이 SSL 표현만 사용한 경우(SSL only)를 비교하였다.
-   **PaCMAP 시각화 결과**: SSL-only 모델은 일부 키워드만 분리되는 반면, meta-only 모델은 클러스터가 명확하지 않았다. 하지만 두 방법을 결합한 **meta+SSL** 모델은 각 키워드의 경계가 매우 뚜렷하고 동일 키워드의 포인트들이 밀집되어 나타났다.
-   이는 SSL이 제공하는 풍부한 일반적 특징 표현이 Meta-learning의 빠른 적응 능력과 결합되어, 처음 보는 키워드에 대해서도 매우 변별력 있는 임베딩 공간을 형성함을 의미한다.

### 비판적 해석 및 한계
-   **모델 크기의 영향**: CPC와 TERA의 성능이 낮게 나타난 점에 대해, 저자들은 모델의 파라미터 크기가 작아 표현력에 한계가 있었을 것으로 추측하고 있다.
-   **계산 비용**: MAML 등의 알고리즘에서 2차 미분 계산 비용을 줄이기 위해 First-Order approximation (FOMAML)을 사용하였으나, 여전히 Meta-learning의 학습 과정은 일반적인 전이 학습보다 복잡할 수 있다.

## 📌 TL;DR

본 논문은 사용자 정의 키워드 감지(User-defined KWS)라는 Few-shot 학습 문제를 해결하기 위해 SSL과 Meta-learning의 결합 효율성을 분석하였다. 실험을 통해 **HuBERT 인코더와 Matching network 알고리즘의 조합**이 가장 높은 정확도와 강건성을 보임을 확인하였다. 특히, SSL의 일반적 특징 학습과 Meta-learning의 빠른 적응 능력이 상호 보완적으로 작용하여 새로운 키워드에 대한 변별력을 극대화한다는 점을 입증하였으며, 이는 향후 개인화된 음성 인식 시스템 구축에 중요한 가이드라인을 제공한다.