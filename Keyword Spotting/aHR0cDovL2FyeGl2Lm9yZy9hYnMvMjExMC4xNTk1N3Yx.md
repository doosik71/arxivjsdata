# Visual Keyword Spotting with Attention

K R Prajwal, Liliane Momeni, Triantafyllos Afouras, Andrew Zisserman (2021)

## 🧩 Problem to Solve

본 논문은 무음 비디오 시퀀스에서 특정 음성 키워드를 찾아내는 **Visual Keyword Spotting (VKWS)** 문제를 해결하고자 한다. 일반적인 Visual Speech Recognition (VSR)이 비디오 내의 모든 단어를 텍스트로 전사(transcription)하는 것과 달리, VKWS는 사용자가 지정한 특정 키워드가 비디오의 어느 시점에 존재하는지를 검출하고 위치를 특정하는 것이 목표이다.

이 문제는 다음과 같은 이유로 중요하다. 첫째, "wake word" 인식과 같이 긴 입력 시퀀스에서 특정 단어만 식별하면 되는 실제 응용 사례가 많기 때문이다. 둘째, 기존의 VSR 방식은 언어 모델(language modelling)에 크게 의존하여 문맥이 제한적이거나 입력 일부가 가려진 경우 성능이 급격히 저하되는 한계가 있다. 따라서 본 연구는 키워드라는 추가 정보를 활용함으로써 VSR보다 더 효율적이고 강건한(robust) 인식 시스템을 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비디오의 시각적 정보와 키워드의 음성적(phonetic) 정보를 통합하여 처리하는 **Transpotter** 아키텍처를 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Transpotter 아키텍처 제안**: 비디오 인코딩 스트림과 키워드의 음성 인코딩 스트림 사이의 전체 교차 모달 어텐션(full cross-modal attention)을 사용하는 Transformer 기반 모델을 설계하였다.
2.  **SOTA 성능 달성**: LRW, LRS2, LRS3와 같은 챌린징한 데이터셋에서 기존의 Visual KWS 및 VSR 방법론들을 큰 차이로 능가하는 성능을 입증하였다.
3.  **극한 환경에서의 적용 가능성 확인**: 수어(sign language) 비디오에서 입모양만으로 단어를 말하는 'mouthings'를 검출하는 매우 어려운 작업에서도 높은 성능을 보여줌으로써 모델의 범용성을 증명하였다.

## 📎 Related Works

**1. Keyword Spotting (KWS)**
전통적인 오디오 KWS는 HMM, Dynamic Time Warping 등을 사용해 왔으며, 최근에는 RNN, CNN, Transformer 기반의 딥러닝 모델로 발전하였다. Visual KWS의 경우 query-by-example, sliding window classification, 혹은 입술 읽기 특징 시퀀스에서 음성 쿼리를 찾는 방식 등이 제안되었다. 본 논문은 이러한 기존 방식들의 약점을 보완하여 더 강력한 비디오-텍스트 모델링과 명시적인 키워드 위치 추정(localization)을 제공한다.

**2. Lip Reading (VSR)**
최근 VSR은 대규모 데이터셋과 Transformer 아키텍처의 도입으로 비약적인 발전을 이루었으며, 일부 모델은 인간의 능력을 상회하는 수준에 도달했다. 하지만 앞서 언급했듯 VSR은 전사 작업의 특성상 언어 모델 의존도가 높다는 한계가 있으며, 이는 단일 키워드를 찾는 KWS 작업과는 차별점이 된다.

**3. Visual Grounding 및 Transformers**
비디오 내에서 자연어 쿼리에 해당하는 구간을 찾는 Visual Grounding 작업과 유사성이 있으며, 본 연구는 시퀀스 모델링 능력이 뛰어나고 어텐션을 통해 위치 특정(localization)이 가능한 Transformer를 기본 블록으로 채택하였다.

## 🛠️ Methodology

### 전체 시스템 구조
Transpotter는 비디오 프레임 시퀀스와 텍스트 키워드를 입력으로 받아, 키워드의 존재 여부(classification)와 해당 위치(localization)를 동시에 예측한다.

### 1. 모달리티별 인코딩 (Uni-modal Encoders)
-   **Text Representation**: 키워드 $q$는 발음 사전(pronunciation dictionary)을 통해 음소(phoneme) 시퀀스로 변환된다. 학습 가능한 임베딩 벡터 $Q \in \mathbb{R}^{n_p \times d}$에 사인파 위치 인코딩(sinusoidal positional encoding, PE)을 더한 후, $N_t$ 레이어의 Transformer Encoder를 통과시켜 $Q^{enc}$를 생성한다.
    $$Q^{enc} = \text{encoder}_q(Q + PE_{1:n_p}) \in \mathbb{R}^{n_p \times d}$$
-   **Video Representation**: 사전 학습된 visual front-end(CNN 또는 VTP)를 통해 각 프레임에서 특징 벡터 $V \in \mathbb{R}^{T \times d}$를 추출한다. 여기에 PE를 더하고 $N_v$ 레이어의 Transformer Encoder를 통과시켜 $V^{enc}$를 생성한다.
    $$V^{enc} = \text{encoder}_v(V + PE_{1:T}) \in \mathbb{R}^{T \times d}$$

### 2. 통합 멀티모달 표현 (Joint Video-Text Representation)
두 스트림의 결과물을 시간 축으로 연결(concatenate)하고, 맨 앞에 학습 가능한 $[CLS]$ 토큰을 추가하여 하나의 시퀀스 $J$를 구성한다.
$$J = ([CLS]; V^{enc}; Q^{enc}) \in \mathbb{R}^{(1+T+n_p) \times d}$$
이후 $N_m$ 레이어의 Joint Transformer Encoder를 통해 비디오와 음소 벡터 간의 관계를 학습하여 최종 표현 $Z$를 얻는다.
$$Z = \text{encoder}_{vq}(J + PE_{1:(1+T+n_p)}) \in \mathbb{R}^{(1+T) \times d}$$

### 3. 예측 헤드 (Prediction Heads)
-   **Classification**: $[CLS]$ 토큰의 출력 벡터 $Z_1$을 MLP 헤드 $f_c$와 시그모이드 함수 $\sigma$에 통과시켜 키워드의 존재 확률 $\hat{y}_{cls}$를 예측한다.
    $$\hat{y}_{cls} = \sigma(f_c(Z_1)) \in \mathbb{R}^1$$
-   **Localisation**: 비디오 프레임에 해당하는 출력 상태 $Z_{2:(T+1)}$를 공유 MLP 헤드 $f_l$에 통과시켜 각 프레임 $t$가 키워드 발화 부분일 확률 $\hat{y}_{loc}$를 예측한다.
    $$\hat{y}_{loc} = \sigma(f_l(Z_{2:(T+1)})) \in \mathbb{R}^T$$

### 4. 학습 절차 및 손실 함수
모델은 binary cross-entropy (BCE) 손실 함수를 사용하여 학습한다.
-   **분류 손실**: $L_{cls} = -\mathbb{E} \text{BCE}(y_{cls}, \hat{y}_{cls})$
-   **위치 손실**: 키워드가 존재할 때만 계산하며, 프레임 단위로 BCE를 적용하여 평균을 낸다.
    $$L_{loc} = -\mathbb{E} \left[ y_{cls} \left( \frac{1}{T} \sum_{t=1}^T \text{BCE}(y_{loc}^t, \hat{y}_{loc}^t) \right) \right]$$
-   **최종 손실**: 두 손실을 하이퍼파라미터 $\lambda$로 가중 합산한다.
    $$L = \lambda L_{cls} + (1-\lambda) L_{loc}$$

## 📊 Results

### 실험 설정
-   **데이터셋**: LRS2 (BBC 방송), LRS3 (TED/TEDx), LRW (단일 단어 클립), BSL Corpus (수어 비디오).
-   **지표**: 분류 성능은 $Acc_{Cls}@k$ 및 $mAP_{Cls}$로 측정하고, 위치 특정 성능은 Intersection-over-Union (IoU) 기준 $\tau=0.5$일 때의 $mAP_{Loc}$로 측정한다.

### 주요 결과
-   **SOTA 달성**: LRS2와 LRS3 데이터셋에서 기존 KWS-Net 및 VSR 베이스라인보다 월등한 성능을 보였다. 특히 VTP 아키텍처를 visual backbone으로 사용했을 때 성능이 더욱 향상되었다.
-   **강건성**: LRW 데이터셋의 경우, Transpotter는 해당 데이터로 학습하지 않았음에도 불구하고 기존 SOTA인 KWS-Net보다 훨씬 높은 정확도($Acc_{Cls}@1$: 85.8% vs 66.6%)를 기록하였다.
-   **수어 mouthings 검출**: BSL Corpus 실험에서 KWS-Net 대비 $mAP_{Cls}$가 15.6에서 29.6으로 크게 상승하여, 도메인 차이가 큰 환경에서도 효과적임을 입증하였다.
-   **분석 결과**: 키워드의 음소 길이($n_p$)가 길수록, 그리고 주변 시각적 문맥(context)이 많을수록 성능이 향상되는 경향을 보였다.

## 🧠 Insights & Discussion

**1. 강점 및 해석**
-   **강력한 감독 학습**: 단순한 분류가 아니라 프레임 단위의 위치 정보를 함께 학습시킴으로써, 결과적으로 분류 성능($mAP_{Cls}$)까지 향상되는 시너지 효과가 확인되었다.
-   **교차 모달 어텐션의 효율성**: Late-fusion 방식과 달리, 모든 레이어에서 비디오 특징과 음소 토큰이 서로를 참조할 수 있게 하여 더 정밀한 매칭이 가능해졌다.

**2. 한계 및 비판적 해석**
-   **Homophemes 문제**: 'mark', 'bark', 'park'와 같이 입모양은 동일하지만 소리가 다른 단어(homophemes)들의 경우, 모델이 이를 구분하지 못하고 모두 동일한 위치에서 검출하는 한계가 있다. 이는 오직 시각 정보만을 사용하기 때문에 발생하는 근본적인 문제이며, 향후 텍스트의 의미론적(semantic) 정보나 주변 문맥 정보를 더 깊게 활용해야 해결 가능할 것으로 보인다.
-   **데이터 의존성**: 수어 비디오의 경우 일부 단어가 부분적으로만 발음(partially mouthed)되거나 손에 의해 가려지는 경우가 있어, 완벽한 검출에는 여전히 어려움이 있다.

## 📌 TL;DR

본 논문은 비디오와 텍스트(음소) 스트림 간의 **전체 교차 모달 어텐션**을 사용하는 Transformer 기반의 **Transpotter** 아키텍처를 제안하여 Visual Keyword Spotting 성능을 혁신적으로 향상시켰다. 특히 프레임 단위의 명시적 위치 추정 학습을 통해 분류 성능까지 끌어올렸으며, 수어 비디오의 입모양 검출과 같은 실제 응용 분야에서도 높은 가능성을 보여주었다. 향후 연구에서는 시각적 모호성을 해결하기 위해 단어의 의미론적 정보와 주변 문맥을 통합하는 방향으로 발전할 것으로 기대된다.