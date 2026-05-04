# Multi-Task Network for Noise-Robust Keyword Spotting and Speaker Verification using CTC-based Soft VAD and Global Query Attention

Myunghun Jung, Youngmoon Jung, Jahyun Goo, and Hoirin Kim (2020)

## 🧩 Problem to Solve

본 논문은 키워드 검출(Keyword Spotting, KWS)과 화자 검증(Speaker Verification, SV)이라는 두 가지 작업을 동시에 수행하는 멀티태스크 네트워크를 제안한다. 일반적으로 KWS와 SV는 독립적인 작업으로 다루어져 왔으나, 음향 도메인과 화자 도메인은 서로 상호 보완적인 관계에 있다.

특히, 본 연구는 다음과 같은 세 가지 도전적인 상황에서의 성능 향상을 목표로 한다.
1. **소음 환경(Noisy Environments):** 배경 소음으로 인해 음성 신호가 왜곡되는 상황.
2. **개방형 어휘 KWS(Open-vocabulary KWS):** 사용자가 임의로 정의한 키워드를 인식해야 하는 상황.
3. **단시간 화자 검증(Short-duration SV):** 매우 짧은 길이의 발화만으로 화자의 신원을 확인해야 하는 상황.

결과적으로 본 논문의 목표는 음향, 화자, 음소 정보를 통합적으로 활용하여, 소음이 많은 환경에서도 짧은 발화만으로 키워드 인식과 화자 검증을 동시에 정밀하게 수행하는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 KWS와 SV의 상호 연관성을 이용하여, 각 작업의 특성 정보를 공유하고 결합함으로써 단독 모델보다 강건한 임베딩을 생성하는 것이다. 이를 위해 다음과 같은 핵심 기술을 도입하였다.

- **CTC 기반 Soft VAD (Connectionist Temporal Classification-based soft Voice Activity Detection):** CTC 손실 함수를 통해 학습된 음소 정보를 활용하여, 음성 신호에서 음소적으로 중요한 프레임을 식별하는 가중치를 생성한다.
- **Global Query Attention (GQA):** 음향 정보와 화자 정보를 통합한 글로벌 쿼리를 생성하고, 이를 통해 각 도메인에 최적화된 특징을 집계(aggregation)함으로써 판별력이 높은 임베딩을 추출한다.
- **멀티태스크 아키텍처:** 강화 네트워크(Enhancement Network), 음향 특징 추출 네트워크, 화자 특징 추출 네트워크, 그리고 풀링 네트워크를 유기적으로 결합하여 상호 기여하도록 설계하였다.

## 📎 Related Works

### 관련 연구 및 한계
- **Open-vocabulary KWS:** 임의의 키워드를 처리하기 위해 고정 차원의 음향 단어 임베딩(Acoustic Word Embeddings)을 사용하며, 주로 Triplet loss나 CTC 기반의 음소 정보를 활용하는 방식이 연구되었다. 하지만 여전히 환경 변화에 취약한 한계가 있다.
- **Speaker Verification:** 화자 식별을 위해 Speaker Embedding을 추출하며, 최근에는 ResNet 기반의 구조가 많이 사용된다. 특히 단시간 SV의 경우, 발화 길이가 짧을수록 정보량이 부족하여 성능이 급격히 저하되는 문제가 있으며, 이를 해결하기 위해 다양한 풀링(pooling) 기법이 제안되었다.

### 기존 방식과의 차별점
기존 연구들이 KWS와 SV를 독립적으로 처리하거나 단순히 개별 모델의 성능 향상에 집중했다면, 본 논문은 두 도메인의 상호 보완적 특성을 이용해 하나의 멀티태스크 네트워크로 통합하였다. 특히 음소 정보를 활용한 Soft VAD와 이를 기반으로 한 Global Query Attention을 통해 짧은 발화에서도 효율적인 정보 집계가 가능하도록 설계한 점이 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 구조
시스템은 크게 네 가지 서브 네트워크로 구성된다: **Enhancement $\rightarrow$ Acoustic Feature Extraction $\rightarrow$ Speaker Feature Extraction $\rightarrow$ Pooling**.

#### 1. Enhancement Network
소음 제거를 통해 KWS와 SV 성능을 동시에 높이기 위해 공유 네트워크로 배치되었다. 두 개의 Dilated CNN이 캐스케이드 구조로 연결되어 있으며, 입력 스펙트로그램 $X$에서 spectral distortion을 추정하여 이를 뺀 $\hat{X}$를 강화된 스펙트로그램으로 출력한다.

#### 2. Acoustic Feature Extraction Network
- **구조:** 두 층의 Bi-directional LSTM ($\text{LSTM}_1^w, \text{LSTM}_2^w$)을 통해 프레임 수준의 음향 특징 $H_c, H_w \in \mathbb{R}^{T \times 512}$를 추출한다.
- **Soft VAD 생성:** $H_c$를 Linear layer와 Log-softmax에 통과시켜 CTC 레이블 시퀀스 $\pi$에 대한 확률 $P_c$를 구한다. 이후 1층의 Bi-LSTM ($\text{LSTM}_c$)과 Sigmoid 함수를 거쳐, 음소적으로 중요한 프레임을 나타내는 Soft VAD 포스테리어 $v_c \in \mathbb{R}^T$를 생성한다.

#### 3. Speaker Feature Extraction Network
- **Phonetic Conditioning:** 음향 네트워크에서 추출된 bottleneck state $Z$를 phonetic conditional vector로 사용하여 $\hat{X}$와 결합한다. 이는 짧은 발화에서도 불필요한 음소 변동을 억제하고 화자 고유의 특징을 잡는 데 도움을 준다.
- **구조:** 6개의 수정된 ResNet 블록($\text{ResCNN}_1 \sim \text{ResCNN}_6$)을 통해 프레임 수준의 화자 특징 $H_s \in \mathbb{R}^{T \times 256}$를 추출한다.

#### 4. Pooling Network (Global Query Attention)
가장 핵심적인 부분으로, 다음의 4단계 과정을 거쳐 최종 임베딩을 생성한다.
1. **Temporary Queries 생성:** Soft VAD 가중치 $v_c$를 사용하여 음향 및 화자 특징의 가중 합을 구한다.
   $$q_g = [q_w, q_s] = \left[ \frac{\sum_{t=1}^T H_w^t v_c^t}{\sum v_c^t}, \frac{\sum_{t=1}^T H_s^t v_c^t}{\sum v_c^t} \right] \in \mathbb{R}^{d_w + d_s}$$
2. **Global Query 생성:** 위에서 생성된 $q_w, q_s$를 결합하여 $q_g$를 형성한다.
3. **Domain Queries 재계산:** $q_g$를 각각 $\text{Linear}_w, \text{Linear}_s$에 통과시켜 각 도메인에 특화된 쿼리 $q^*_w, q^*_s$를 생성한다.
4. **Aggregation:** Multi-head Attention을 통해 최종 임베딩 $e_w$(단어 임베딩)와 $e_s$(화자 임베딩)를 도출한다.
   $$e_k = \text{Multi-Head}_k(q^*_k, H_k, H_k), \quad k \in \{w, s\}$$

### 학습 목표 및 손실 함수
전체 네트워크는 다음의 합산 손실 함수를 통해 학습된다.
$$L = L_w + L_s + L_c$$
- $L_w, L_s$: KWS와 SV를 위한 $L_2$-constrained softmax loss이다. (각각 $\|e_w\|=6, \|e_s\|=12$로 제약)
- $L_c$: 음소 정보 학습을 위한 CTC loss이다.
- Enhancement 네트워크는 명시적인 손실 함수 없이 KWS/SV의 성능 향상을 통해 간접적으로 학습된다.

## 📊 Results

### 실험 설정
- **데이터셋:** Google Speech Commands V2 (35개 단어, 2618명 화자). 
- **소음 환경:** MUSAN 데이터셋의 'Music', 'Babble', 'Others' 소음을 섞어 SNR 20, 10, 5, 0dB의 13가지 환경을 구성하였다.
- **평가 지표:** Equal Error Rate (EER)를 사용하며, 1초-1초(등록-테스트)의 단시간 제약을 적용하였다.

### 주요 결과
제안된 멀티태스크 네트워크는 모든 환경에서 베이스라인(개별 학습 모델)보다 우수한 성능을 보였다.
- **KWS:** 평균적으로 상대적 성능 향상 **4.06%** 달성.
- **SV:** 평균적으로 상대적 성능 향상 **26.71%** 달성 (절대적 수치로 2~4% 감소).
- 이는 음향, 음소, 화자 도메인을 통합 학습하는 것이 특히 화자 검증 성능을 크게 끌어올림을 시사한다.

### Ablation Study 결과
- **Enhancement Network 제거 시:** 낮은 SNR 환경에서 KWS 성능이 크게 저하되었다.
- **Phonetic Conditioning 제거 시:** 특히 'Others' 소음 환경에서 SV 성능이 하락하였다.
- **Pooling 방식 비교:** Global Query Attention이 일반적인 Self-attention보다 우수한 성능을 보였다.
- **CTC 모듈 전체 제거 시:** 성능이 유의미하게 하락하여, 음소 정보의 상호 기여가 Enhancement 네트워크의 존재만큼이나 중요함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 KWS와 SV가 서로 다른 작업임에도 불구하고, 음소 정보($v_c$)를 매개체로 활용하여 두 작업의 시너지를 냈다는 점이 매우 인상적이다. 시각화 결과(Fig 3)를 통해 Enhancement 네트워크가 단순히 소음을 지우는 것이 아니라 하모닉스 성분을 강조하여 주파수 해상도를 높인다는 점과, CTC-based soft VAD가 실제 음성 구간에서 높은 포스테리어를 가져 효율적인 쿼리 생성을 가능케 함을 확인하였다.

### 한계 및 비판적 해석
- **Babble Noise의 모순:** Ablation 결과에서 'Babble' 소음(배경에 다른 사람의 목소리가 섞인 소음) 환경에서는 음소 정보가 오히려 혼란을 주어 SV 성능에 부정적인 영향을 미칠 수 있음이 나타났다. 이는 배경 음성으로 인해 잘못된 음소 가중치가 생성되어 쿼리가 오염되었기 때문으로 해석된다.
- **데이터셋의 한계:** Google Speech Commands 데이터셋은 발화 길이가 1초 이하로 매우 짧다. 실제 환경에서는 더 긴 발화가 포함될 수 있는데, 이때도 GQA 방식이 기존의 통계적 풀링보다 우월할지는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 **CTC 기반 Soft VAD**와 **Global Query Attention**을 도입한 **멀티태스크 네트워크**를 통해 소음 환경, 개방형 어휘 KWS, 단시간 SV 문제를 동시에 해결하였다. 특히 음향-음소-화자 정보를 통합하여 학습함으로써 화자 검증(SV)에서 26.71%의 높은 상대적 성능 향상을 이끌어냈다. 이 연구는 음성 인식의 전단 단계(Wake-up word 및 사용자 인증)에서 두 작업을 통합 처리함으로써 시스템의 효율성과 강건성을 동시에 확보할 수 있는 가능성을 제시하였다.