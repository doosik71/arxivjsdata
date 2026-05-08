# ORTHOGONALITY CONSTRAINED MULTI-HEAD ATTENTION FOR KEYWORD SPOTTING

Mingu Lee, Jinkyu Lee, Hye Jin Jang, Byeonggeun Kim, Wonil Chang and Kyuwoong Hwang (2019)

## 🧩 Problem to Solve

본 논문은 음성 인식 기반의 키워드 스포팅(Keyword Spotting, KWS) 시스템에서 Multi-head Attention(MHA) 메커니즘을 사용할 때 발생하는 **헤드 간의 중복성(Redundancy)** 문제를 해결하고자 한다.

Multi-head Attention은 이론적으로 시퀀스의 서로 다른 부분(예: 음절, 단어 조각)에 주목하여 풍부한 표현을 학습할 수 있다. 그러나 제약 조건이 없는 일반적인 MHA를 그대로 사용할 경우, 각 어텐션 헤드가 서로 유사한 위치에 주목하거나 유사한 표현을 생성하는 positional 및 representational 중복성이 발생하여 네트워크의 효율성이 떨어진다.

따라서 본 연구의 목표는 각 어텐션 헤드가 서로 다른 하위 시퀀스에서 상호 배타적이고 다양한 정보를 추출하도록 강제하는 **직교성 제약(Orthogonality Constraints)** 기반의 정규화 기법을 제안하여 KWS 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 어텐션 헤드들 사이의 **직교성(Orthogonality)**과 **비직교성(Non-orthogonality)**을 정규화 항으로 추가하여, MHA가 음성 신호의 구조적 정보를 효과적으로 캡처하도록 유도하는 것이다.

1. **Inter-head Orthogonality**: 서로 다른 헤드 간의 Context vector와 Score vector가 직교하도록 하여, 각 헤드가 서로 다른 시간적 위치와 표현 공간을 학습하게 한다.
2. **Intra-head Non-orthogonality**: 동일한 헤드라면 서로 다른 샘플에 대해서도 유사한 Context vector를 생성하도록 하여, 특징 공간에서의 변동성을 줄이고 분류 성능을 높인다.
3. **Selective Regularization**: 위 제약 조건들을 모든 데이터가 아닌, 키워드가 포함된 **양성 샘플(Positive samples)**에만 선택적으로 적용하여 학습의 효율성을 높인다.

## 📎 Related Works

기존의 키워드 스포팅 연구들은 주로 다음과 같은 접근 방식을 취했다.

- **HMM 기반 접근**: Hidden Markov Model(HMM)을 사용하여 키워드와 일반 음성(filler)의 음향적 특성을 명시적으로 모델링하였다. 이후 GMM이 딥러닝 아키텍처(DNN, CNN, CRNN)로 대체되었으나, 여전히 정교한 시간 정렬 라벨(Time-aligned labels)이 필요하다는 한계가 있다.
- **Attention 기반 접근**: 최근에는 end-to-end 구조의 어텐션 모델이 제안되었으나, 대부분 단일 헤드(Single-head) 어텐션을 사용하여 전체 시퀀스를 하나의 컨텍스트 벡터로 요약하는 수준에 그쳤다.
- **MHA 및 Disagreement Regularization**: Machine Translation 분야에서 MHA의 다양성을 위해 Cosine similarity 기반의 disagreement regularization이 제안된 바 있다. 본 논문은 이러한 개념을 음성 데이터와 KWS 작업의 특성에 맞게 최적화하여 적용하였다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

시스템은 Encoder, Attention, Classifier의 세 단계로 구성된다.

- **Encoder**: 40차원의 Mel-filter bank 에너지를 입력으로 받으며, CNN 레이어(kernel size $5 \times 20$, stride $2 \times 1$)와 GRU 레이어(64 hidden units)로 구성된 CRNN 구조를 사용하여 스펙트럼 및 시간적 특성을 추출하여 hidden representation $h[t]$를 생성한다.
- **Attention**: $H$개의 어텐션 헤드가 각각 독립적으로 컨텍스트 벡터 $c_i$를 생성하고, 이를 최종적으로 연결(concatenate)하여 전체 컨텍스트 벡터 $c$를 형성한다.
- **Classifier**: 연결된 벡터 $c$에 선형 변환과 Softmax를 적용하여 키워드 존재 확률 $p(y|x)$를 계산하는 이진 분류를 수행한다.

### 2. 기본 어텐션 메커니즘

각 헤드 $i$에서의 어텐션 가중치 $\alpha_i[t]$는 다음과 같이 계산된다.
$$\alpha_i[t] = \frac{\exp(e_i[t])}{\sum_{\tau=1}^{T} \exp(e_i[\tau])} \quad (1)$$
여기서 스칼라 점수 $e_i[t]$는 다음과 같은 비선형 함수로 계산된다.
$$e_i[t] = v_i^T \tanh(W_i h[t] + b_i) \quad (2)$$
최종 컨텍스트 벡터 $c_i$는 가중 합으로 계산된다.
$$c_i = \sum_{t=1}^{T} \alpha_i[t] h[t] \quad (3)$$

### 3. 직교성 정규화 (Orthogonality Regularization)

본 논문은 다음의 세 가지 정규화 항을 도입한다.

**가. Inter-head Orthogonality ($\mathcal{L}_{inter}^c, \mathcal{L}_{inter}^s$):**
헤드 간의 컨텍스트 벡터 $c_i$와 점수 벡터 $e_i$가 서로 직교하도록 하여 중복성을 제거한다. Frobenius norm을 사용하여 정규화 항을 정의한다.
$$\mathcal{L}_{inter}^c = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{H(H-1)} \| C^{(n)T} C^{(n)} - I_H \|_F^2 \quad (4)$$
$$\mathcal{L}_{inter}^s = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{H(H-1)} \| E^{(n)T} E^{(n)} - I_H \|_F^2 \quad (5)$$
여기서 $C^{(n)}$과 $E^{(n)}$은 정규화된 컨텍스트 및 점수 벡터들로 구성된 행렬이다.

**나. Intra-head Non-orthogonality ($\mathcal{L}_{intra}^c$):**
동일한 헤드 $i$가 서로 다른 양성 샘플들에 대해 유사한 컨텍스트 벡터를 생성하도록 유도한다.
$$\mathcal{L}_{intra}^c = \frac{1}{H} \sum_{i=1}^{H} \frac{1}{N(N-1)} \| \tilde{C}_i^T \tilde{C}_i - I_N \|_F^2 \quad (8)$$
여기서 $\tilde{C}_i$는 모든 샘플의 $i$번째 헤드 컨텍스트 벡터를 모은 행렬이다.

**다. 선택적 정규화 및 최종 손실 함수:**
위 제약들은 양성 샘플($y^{(n)}=1$)에 대해서만 유효하므로, 선택 행렬 $Y$를 도입하여 $\tilde{\mathcal{L}}$ 형태로 수정 적용한다. 최종 목적 함수는 다음과 같다.
$$\theta^* = \arg\min_{\theta} \{ \mathcal{L}_{CE} + \lambda_1 \tilde{\mathcal{L}}_{inter}^c - \lambda_2 \tilde{\mathcal{L}}_{intra}^c + \lambda_3 \tilde{\mathcal{L}}_{inter}^s \} \quad (13)$$
$\mathcal{L}_{intra}^c$는 최대화해야 하므로 마이너스($-$) 부호를 갖는다.

## 📊 Results

### 1. 실험 설정

- **대상 키워드**: "Hey Snapdragon" (4음절)
- **데이터셋**: 325명의 화자로부터 수집된 양성/음성 샘플. 검증 및 테스트 셋에는 4종의 노이즈(babble, car, music, office)와 잔향(reverberation)을 추가하여 강건성을 평가하였다.
- **평가 지표**: 1 FA/hr (시간당 오경보 1회) 임계값에서의 FRR(False Rejection Rate, 오거절률).

### 2. 주요 결과

- **정규화 효과**: 정규화 기법을 적용했을 때 FRR이 유의미하게 감소하였다.
- **비교 결과 (at 1 FA/hr)**:
  - Single-head 모델 대비 FRR 최대 **34.4%** 감소.
  - Plain Multi-head(정규화 없음) 모델 대비 FRR 최대 **36.0%** 감소.
- **정규화 조합**: Table 1에 따르면 모든 정규화 항($\lambda_1, \lambda_2, \lambda_3$)을 동시에 적용했을 때 가장 낮은 FRR을 기록하였다.
- **하이퍼파라미터 $\lambda$**: $\lambda=0.1$일 때 최적의 성능을 보였다.

## 🧠 Insights & Discussion

**강점 및 효과:**

- **구조적 정보 캡처**: 본 논문의 제안 방법은 명시적인 시퀀스 모델(HMM 등) 없이도 MHA가 음성 신호의 부분적인 구조(예: 음절 단위)를 효율적으로 학습하게 만든다.
- **시각적 분석**: Attention weight의 시각화 결과(Fig 2), 정규화가 없는 MHA는 헤드 간의 주목 영역이 겹치는 경향이 있으나, 제안된 방법을 적용하면 각 헤드가 서로 배타적인 영역에 주목하는 것이 확인되었다. 이는 모델이 중복 없이 풍부한 특징을 추출하고 있음을 시사한다.
- **특징 일관성**: Intra-head non-orthogonality 제약을 통해 양성 샘플들 간의 특징 표현 변동성을 줄임으로써 분류기의 결정 경계를 더 명확하게 만들 수 있었다.

**한계 및 논의:**

- **데이터 의존성**: 본 실험은 "Hey Snapdragon"이라는 특정 키워드에 집중되어 있으며, 키워드의 길이나 음절 수에 따라 최적의 헤드 수($H$)나 $\lambda$ 값이 달라질 수 있다.
- **계산 비용**: 모델 사이즈는 약간 증가하지만(4-head 기준 91k vs 78k), 실시간 KWS 시스템에서 이 정도의 파라미터 증가가 실제 추론 속도에 미치는 영향에 대한 분석은 부족하다.

## 📌 TL;DR

본 논문은 Multi-head Attention 기반의 키워드 스포팅 시스템에서 발생하는 헤드 간 중복성 문제를 해결하기 위해 **직교성 제약 기반의 정규화 기법**을 제안하였다. 헤드 간에는 직교성을 부여하여 서로 다른 정보를 추출하게 하고, 동일 헤드의 샘플 간에는 유사성을 부여하여 특징의 일관성을 높였다. 이를 통해 "Hey Snapdragon" 키워드 인식 성능을 크게 향상시켰으며, 이는 MHA가 음성 데이터의 구조적 특징을 효율적으로 학습하는 효과적인 방법임을 입증하였다. 향후 화자 확인(Speaker Verification)이나 일반 음성 인식 작업으로의 확장 가능성이 높다.
