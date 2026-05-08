# SELF-SUPERVISED SPEECH REPRESENTATION LEARNING FOR KEYWORD-SPOTTING WITH LIGHT-WEIGHT TRANSFORMERS

Chenyang Gao, Yue Gu, Francesco Caliva, Yuzong Liu (2023)

## 🧩 Problem to Solve

본 논문은 리소스가 제한된 온디바이스(On-device) 환경에서 Keyword-Spotting (KS) 성능을 향상시키기 위해 Self-supervised speech representation learning (S3RL)을 적용하는 문제를 다룬다. 일반적으로 S3RL 연구들은 수백만 개의 파라미터를 가진 거대 모델을 사용하지만, 실제 모바일이나 에지 디바이스의 메모리 및 연산 제약으로 인해 이러한 대형 모델을 그대로 적용하는 것은 불가능하다.

또한, 기존의 S3RL 방법론들은 주로 자동 음성 인식(Automatic Speech Recognition, ASR) 작업에 최적화되어 설계되었다. ASR과 달리 KS, 감정 분석, 화자 식별과 같은 분류(Classification) 작업에서는 발화 단위의 차이(Utterance-wise difference)를 학습하는 것이 매우 중요함에도 불구하고, 기존 S3RL은 이러한 발화 수준의 표현 학습을 간과하고 있다는 점이 문제로 지적된다. 따라서 본 논문의 목표는 매우 경량화된 Transformer 모델에서도 효과적으로 동작하는 S3RL 프레임워크를 제안하고, 특히 분류 성능을 높이기 위한 발화 단위 구분 능력을 강화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **경량 Transformer를 통한 S3RL의 가능성 입증**: 단 330K 개의 파라미터만을 가진 경량 Transformer 모델에 S3RL을 적용하여, 온디바이스 KS 작업에서도 사전 학습(Pre-training)이 효과적임을 증명하였다.
2. **발화 단위 구분 강화 메커니즘(Utterance-wise distinction boosting) 제안**: 모든 S3RL 방법론에 쉽게 결합할 수 있는 2단계 대조 학습(Contrastive Learning) 구조를 제안하여, 모델이 발화 간의 추상적인 차이를 더 잘 학습하게 함으로써 KS 성능을 향상시켰다.
3. **경량 모델 대상 S3RL 방법론의 비교 분석**: APC, MPC, CL 등 다양한 S3RL 접근 방식을 동일한 경량 모델 벤치마크에서 비교하여, 어떤 방법이 경량 모델의 사전 학습에 가장 적합한지에 대한 학술적 근거를 제시하였다.

## 📎 Related Works

논문에서는 세 가지 주요 S3RL 접근 방식을 소개한다.

- **Auto-regressive Predictive Coding (APC)**: 과거의 타임스탬프 정보를 사용하여 미래의 프레임을 예측하는 생성적 손실 기반 방식이다.
- **Masked Predictive Coding (MPC)**: 양방향 컨텍스트를 활용하여 마스킹된 특징을 복원하는 생성적 손실 기반 방식이다.
- **Contrastive Learning (CL)**: 긍정 샘플 쌍과 부정 샘플 쌍 사이의 거리를 최대화하는 판별적 학습 방식이다.

기존 연구들은 주로 거대 모델을 사용하여 고품질의 표현을 생성하는 데 집중했으나, 본 논문은 이를 경량 모델로 확장했다는 점에서 차별점을 가진다. 또한, 기존 S3RL이 주로 ASR을 겨냥해 프레임 수준의 표현에 집중한 반면, 본 연구는 KS를 위해 발화 수준(Utterance-level)의 표현력을 높이는 메커니즘을 추가하였다.

## 🛠️ Methodology

### 1. Light-weight Transformer 구조

연산 효율성을 위해 다음과 같은 설계를 적용하였다.

- **입력 처리**: VGG-like 모델과 Strided-convolution 기반의 2배 다운샘플링을 통해 입력 시퀀스 길이를 줄여 Transformer의 $T^2$ 시간 복잡도 문제를 완화하였다.
- **모델 크기**: 깊이와 너비를 줄여 전체 파라미터 수를 330K 개로 제한하였다.
- **수정 사항**: APC에서는 과거 정보만 참조하도록 Attention mask를 도입하였고, MPC에서는 학습 가능한 Mask-embeddings를 사용하여 복원 성능을 높였다.

### 2. S3RL 학습 목적 함수

사전 학습에 사용된 세 가지 손실 함수는 다음과 같다.

- **APC 손실**: 현재 프레임에서 $n$ 스텝 앞의 미래 프레임을 예측한다.
$$L_{APC} = \sum_{i=1}^{T-n} \|x_{i+n} - y_i\|$$
- **MPC 손실**: 마스킹된 영역과 마스킹되지 않은 영역에 가중치 $w$를 두어 복원한다.
$$L_{MPC} = \sum_{i=1}^{T} w_i \|x_i - y_i\|$$
- **CL 손실**: InfoNCE 손실과 코드북의 활용도를 높이는 Diversity loss($L_D$)를 결합하여 사용한다.
$$L_{CL} = -\sum_{i=1}^{T} w_i \log \frac{\exp(\Phi(y_i, q_i)/\kappa)}{\sum_{\tilde{q} \in Q} \exp(\Phi(y_i, \tilde{q})/\kappa)} + \beta L_D$$

### 3. 발화 단위 구분 강화 (Utterance-wise distinction boosting)

본 논문이 제안하는 핵심 메커니즘으로, 두 개의 경량 Transformer($LWT_1, LWT_2$)를 사용한다.

- **구조**: $LWT_1$은 학습 대상 모델이며, $LWT_2$는 동일한 S3RL로 사전 학습된 후 가중치가 고정(Frozen)된 특징 추출기이다.
- **절차**:
    1. 두 모델의 두 번째 레이어에서 특징을 추출하고 Mean-pooling을 통해 발화 수준 표현 $u_1, u_2$를 생성한다.
    2. $u_2$를 앵커(Anchor)로 사용하여 벡터 양자화(Vector Quantization)를 통해 긍정($\tilde{Q}^+$) 및 부정($\tilde{Q}^-$) 샘플 쌍을 생성한다.
    3. $LWT_1$은 S3RL 과제와 동시에 아래의 발화 단위 대조 학습 손실 $L_{utt}$를 최소화하도록 학습된다.
$$L_{utt} = -\log \frac{\exp(\Phi(u_1, \tilde{Q}^+)/\kappa)}{\sum_{\tilde{q} \in Q} \exp(\Phi(u_1, \tilde{q})/\kappa)} + \beta L_D$$

- **최종 학습 목표**: S3RL 손실과 발화 구분 손실을 가중합하여 학습한다 ($\alpha=0.9$).
$$L = \alpha L_{S3RL} + (1-\alpha) L_{utt}$$

## 📊 Results

### 실험 설정

- **데이터셋**: 사전 학습에는 LibriSpeech (960시간)를 사용하였으며, 다운스트림 KS 작업에는 Google Speech Commands v2 (35개 클래스) 및 사내(In-house) 데이터셋을 사용하였다.
- **특징 추출**: 64-D LFBE spectrogram을 사용하였다.
- **평가 지표**: Google 데이터셋에서는 Accuracy를, 사내 데이터셋에서는 고정된 False Rejection Rate (FRR)에서의 상대적 False Acceptance Rate (FAR)를 측정하였다.

### 주요 결과

- **S3RL의 유효성**: 모든 S3RL 방법론이 처음부터 학습(Training from scratch)한 BaselineLT보다 높은 성능을 보였다. 이는 매우 작은 모델에서도 대량의 라벨 없는 데이터를 이용한 사전 학습이 유효함을 시사한다.
- **최적의 방법론**: **APC** 기반의 사전 학습이 Google 및 사내 데이터셋 모두에서 가장 뛰어난 성능을 보였다.
- **제안 방법의 효과**: 발화 단위 구분 강화($+$ 표시)를 적용했을 때 성능이 추가로 향상되었다. 특히 $\text{LTAPC}^+$ 모델은 BaselineLT 대비 Google 데이터셋에서 1.2%의 정확도 향상을 보였으며, 사내 데이터셋의 4개 키워드에 대해 6% ~ 23.7%의 상대적 FAR 개선을 달성하였다.
- **Ablation Study**: APC의 미래 예측 스텝 $n=8$, MPC의 마스크 비율 50%, CL의 코드북 크기 64일 때 최적의 성능이 나타났다.

## 🧠 Insights & Discussion

**강점 및 시사점**
본 연구는 S3RL이 반드시 거대 모델에서만 작동하는 것이 아니며, 적절한 아키텍처 설계와 목적 함수를 통해 330K 수준의 초경량 모델에서도 유의미한 성능 향상을 이끌어낼 수 있음을 보여주었다. 특히, 단순히 프레임 단위의 예측에 그치지 않고 발화 수준의 표현력을 강제하는 대조 학습 메커니즘을 도입함으로써 분류 작업(KS)에 특화된 사전 학습 방향을 제시하였다.

**한계 및 비판적 해석**

- **모델 용량의 한계**: CL 기반 방식이 APC나 MPC보다 성능이 낮게 나온 점은, 대조 학습이 유의미한 표현을 찾기 위해 더 많은 모델 용량(Capacity)을 요구하기 때문이라는 저자의 추측이 타당해 보인다.
- **풀링 방식의 단순함**: 발화 표현을 얻기 위해 Mean-pooling을 사용하였으나, 이는 시퀀스의 시간적 정보를 단순화시킨다는 단점이 있다. 논문에서도 언급되었듯이, 채널 간 상관관계를 분석하는 등 더 정교한 특징 추출 방식에 대한 연구가 필요하다.
- **데이터 편향**: 사내 데이터셋의 경우 키워드별 데이터 양의 차이가 매우 크지만, 이에 대한 세부적인 분석이나 불균형 해소 방안은 명시되지 않았다.

## 📌 TL;DR

본 논문은 온디바이스 Keyword-Spotting을 위해 **330K 파라미터의 초경량 Transformer**에 **S3RL(자기지도 학습)**을 적용하는 방법을 제안한다. 특히 발화 간의 차이를 극대화하는 **2단계 대조 학습 메커니즘**을 도입하여 분류 성능을 높였으며, 실험 결과 **APC(Auto-regressive Predictive Coding)** 방식이 경량 모델 사전 학습에 가장 효율적임을 확인하였다. 이 연구는 리소스가 극도로 제한된 환경에서도 S3RL을 통해 데이터 효율적인 모델 학습이 가능함을 입증하였다.
