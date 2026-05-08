# Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer

Xiaojie Gao, Yueming Jin, Yonghao Long, Qi Dou, and Pheng-Ann Heng (2021)

## 🧩 Problem to Solve

본 논문은 현대 수술실의 안전성과 질을 높이기 위한 지능형 문맥 인식 시스템(Context-Aware Systems, CAS)의 핵심 과제인 수술 단계 인식(Surgical Phase Recognition) 문제를 해결하고자 한다. 수술 단계 인식은 수술 모니터링, 프로토콜 추출 및 의사 결정 지원을 위해 필수적이다.

그러나 비전 기반의 인식은 클래스 간의 유사한 외관(inter-class appearance)과 녹화된 영상의 흐림(blur) 현상으로 인해 매우 까다롭다. 특히 실시간 온라인 인식의 경우, 현재의 결정을 내리기 위해 미래의 정보를 사용할 수 없다는 제약이 있으며, 고차원 비디오 데이터를 실시간으로 처리해야 한다는 시간적 제약이 존재한다.

기존의 많은 연구들이 공간적 특징을 먼저 추출한 후 이를 시간적 특징으로 확장하는 순차적(spatio-temporal order) 구조를 채택해 왔으나, 이 과정에서 중간 단계의 공간적 특징들이 가진 보완적 이점들이 무시되어 중요한 시각적 속성이 손실되는 문제가 발생한다. 따라서 본 논문의 목표는 공간적 특징과 시간적 특징의 시너지를 극대화하여 정확도를 높이면서도 실시간 처리가 가능한 수술 단계 인식 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 워크플로우 분석 분야에 처음으로 Transformer 구조를 도입하여, 무시되었던 공간적 특징과 시간적 특징의 상호 보완적 효과를 재고하는 것이다.

구체적으로, **Hybrid Embedding Aggregation Transformer**를 제안하여 공간적 임베딩(Spatial Embedding)과 시간적 임베딩(Temporal Embedding)을 영리하게 융합한다. 이 구조는 시간적 임베딩 시퀀스에서 공간적 정보를 기반으로 능동적인 쿼리(active queries)를 수행함으로써, 시간적 특징 추출 과정에서 누락되었을 수 있는 세부 공간 정보를 다시 찾아내어 인식 정확도를 높인다. 또한, 하이브리드 임베딩을 병렬로 처리함으로써 매우 높은 추론 속도를 달성하였다.

## 📎 Related Works

기존의 수술 단계 인식 접근 방식은 크게 세 가지 단계로 발전해 왔다.

1. **통계적 모델:** Conditional Random Field(CRF)나 Hidden Markov Models(HMM) 등이 사용되었으나, 수술 프레임 간의 복잡한 시간적 관계를 표현하는 능력이 제한적이었다.
2. **재귀 모델 (RNN/LSTM):** SV-RCNet과 같이 ResNet과 LSTM을 결합하여 엔드-투-엔드 방식으로 시공간 의존성을 모델링했다. 하지만 LSTM은 매우 긴 시퀀스에 대해 기억력의 한계가 있으며, 순차적 계산 방식으로 인해 처리 속도가 느리다.
3. **합성곱 모델 (CNN/TCN):** 3D CNN이나 Temporal Convolutional Networks(TCN)를 사용하여 긴 시간적 관계를 탐색하는 TeCNO와 같은 모델이 제안되었다.

이러한 기존 방식들의 공통적인 한계는 공간적 특징을 추출한 뒤 시간적 특징을 추출하는 순차적 구조를 가진다는 점이다. 이로 인해 최종 결정 단계에서 초기 공간 특징이 가진 세부 정보가 소실되는 경향이 있다. 반면, 본 논문의 Trans-SVNet은 Transformer의 병렬 연산 능력과 어텐션 메커니즘을 통해 시공간 특징을 동시에 고려함으로써 이러한 한계를 극복하고 차별성을 가진다.

## 🛠️ Methodology

Trans-SVNet은 크게 **임베딩 모델(Embedding Model)**과 **집계 모델(Aggregation Model)**의 두 단계로 구성된다.

### 1. 비디오 임베딩 추출 (Video Embedding Extraction)

비디오 프레임을 공간적 임베딩 $l$과 시간적 임베딩 $g$로 표현한다.

- **공간적 임베딩 ($l_t$):** ResNet50을 사용하여 각 프레임 $x_t$에서 변별력 있는 공간 특징을 추출한다. 평균 풀링 레이어의 출력을 통해 $l_t \in \mathbb{R}^{2048}$ 차원의 임베딩을 생성한다.
- **시간적 임베딩 ($g_t$):** 메모리와 시간을 절약하기 위해, 고정된 ResNet50에서 생성된 $l_t$를 입력으로 사용한다. $1 \times 1$ 합성곱 레이어로 차원을 $\mathbb{R}^{32}$로 조정한 뒤, 두 단계의 TCN 모델인 TeCNO를 통과시켜 시간적 임베딩 $g_t \in \mathbb{R}^N$을 생성한다.

### 2. Transformer 레이어 (Transformer Layer)

Transformer 레이어는 Multi-head Attention과 Feed-forward 레이어로 구성된다. 쿼리 $q$와 시간적 시퀀스 $s_{1:n} = [s_1, \dots, s_n]$이 주어졌을 때, 각 헤드는 다음과 같이 어텐션을 계산한다.

$$\text{Attn}(q, s_{1:n}) = \text{softmax}\left(\frac{W^q q (W^k s_{1:n})^T}{\sqrt{d_k}}\right) W^v s_{1:n}$$

여기서 $W$는 선형 매핑 행렬이며, $d_k$는 선형 변환 후 쿼리의 차원이다. 결과물은 잔차 연결(residual connection)과 레이어 정규화(layer normalization)를 거쳐 최종적으로 $\text{Trans}(q, s_{1:n})$으로 출력된다.

### 3. 하이브리드 임베딩 집계 (Hybrid Embedding Aggregation)

집계 모델은 두 개의 Transformer 레이어를 사용하여 최종 예측 $p_t$를 산출한다.

1. **내부 집계 (Internal Aggregation):**
   - 공간적 임베딩 $l_t$는 $\tilde{l}_t = \tanh(W^l l_t)$를 통해 차원이 축소된다.
   - 시간적 임베딩 시퀀스 $g_{t-n+1:t}$는 첫 번째 Transformer 레이어를 통해 자기 주의(self-attention)를 수행하여 $\tilde{g}_{t-n+1:t}$를 생성한다.
   $$\tilde{g}_i = \text{Trans}(g_i, g_{t-n+1:t}), \quad i=t-n+1, \dots, t$$

2. **하이브리드 집계 (Hybrid Aggregation):**
   두 번째 Transformer 레이어에서 $\tilde{l}_t$를 **쿼리(Query)**로, $\tilde{g}_{t-n+1:t}$를 **키(Key) 및 값(Value)**으로 사용하여 시간적 시퀀스에서 필요한 정보를 검색한다.

3. **최종 예측:**
   최종 출력은 Softmax 함수를 통해 단계별 확률로 변환된다.
   $$p_t = \text{Softmax}(\text{Trans}(\tilde{l}_t, \tilde{g}_{t-n+1:t}))$$

학습은 다음과 같은 교차 엔트로피 손실 함수(Cross-Entropy Loss)를 사용한다.
$$\mathcal{L}_C = -\sum_{t=1}^T y_t \log(p_t)$$

## 📊 Results

### 실험 설정

- **데이터셋:** Cholec80(80개 비디오, 7개 단계) 및 M2CAI16(41개 비디오, 8개 단계).
- **평가 지표:** Accuracy (AC), Precision (PR), Recall (RE), Jaccard Index (JA).
- **구현 세부사항:** ResNet50(ImageNet 사전 학습), TeCNO(시간적 임베딩 생성), NVIDIA GeForce RTX 2080 Ti 사용. 시간적 시퀀스 길이 $n=30$, 어텐션 헤드 수 8개.

### 주요 결과

- **정량적 결과:** Trans-SVNet은 비교 대상이 된 7가지 기존 방법론보다 우수한 성능을 보였다. 특히 Cholec80 데이터셋에서 JA 지표가 79.3%로 가장 높게 나타났다.
- **처리 속도:** 저차원 비디오 임베딩 설계를 통해 **91 fps**라는 매우 빠른 추론 속도를 달성하여 실제 수술 영상 기록 속도를 훨씬 상회하는 실시간 성능을 입증했다.
- **정성적 결과:** 시각화 결과, 단순 ResNet은 노이즈가 많고 TeCNO는 매끄럽지만 일부 단계(P2)에서 오분류가 발생했다. Trans-SVNet은 하이브리드 집계를 통해 더 일관되고 강건한(robust) 예측 결과를 보여주었다.

### Ablation Study

- **시퀀스 길이 ($n$):** $n=0$일 때보다 $n>0$일 때 성능이 급격히 향상되며, $n \in [20, 40]$ 범위에서 성능이 안정화됨을 확인하여 $n=30$을 선택했다.
- **아키텍처 구성:** 다양한 쿼리-키 조합을 실험한 결과, **공간 임베딩($l_t$)을 쿼리로 사용하고 시간 임베딩($g$)을 키로 사용하는 구성**이 가장 높은 성능을 보였으며, 이는 통계적으로 유의미함(P-value < 0.05)이 증명되었다.

## 🧠 Insights & Discussion

본 논문은 공간적 특징을 단순히 시간적 특징의 입력으로 사용하는 것에 그치지 않고, 최종 결정 단계에서 **'쿼리'**로 재활용함으로써 시간적 특징 추출 과정에서 손실된 세부 정보를 복원할 수 있음을 보여주었다. 특히 수술 영상에서 흔히 발생하는 빛 반사(reflection)와 같은 노이즈 상황에서, TeCNO와 같은 기존 TCN 기반 모델들이 취약했던 부분을 Transformer의 집계 메커니즘이 효과적으로 보완하여 강건성을 높였다.

또한, 모델의 파라미터 증가량은 매우 적으면서도(약 30k 증가) 성능 향상은 뚜렷하며, 추론 속도가 매우 빠르다는 점은 실제 수술실의 실시간 모니터링 시스템에 적용 가능성이 매우 높음을 시사한다. 다만, M2CAI16보다 Cholec80에서 더 큰 성능 향상이 나타난 것은, 본 모델이 데이터셋의 규모가 크고 복잡도가 높을수록 더 강력한 성능을 발휘하는 특성이 있음을 의미한다.

## 📌 TL;DR

Trans-SVNet은 ResNet(공간)과 TCN(시간) 임베딩을 Transformer를 통해 융합하는 하이브리드 집계 구조를 제안하여 수술 단계 인식의 정확도를 높인 연구이다. 특히 공간 임베딩을 쿼리로 사용하여 시간적 시퀀스에서 누락된 정보를 능동적으로 검색함으로써 SOTA 성능을 달성했으며, 91 fps의 빠른 속도로 실시간 적용 가능성을 입증했다. 이 연구는 시공간 특징의 병렬적 융합이 수술 워크플로우 분석의 핵심임을 보여주었으며, 향후 실시간 수술 지원 시스템의 기반 기술로 활용될 가능성이 크다.
