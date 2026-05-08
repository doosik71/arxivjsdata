# Massive End-to-end Models for Short Search Queries

Weiran Wang, Rohit Prabhavalkar, Dongseong Hwang, Qiujia Li, Khe Chai Sim, Bo Li, James Qin, Xingyu Cai, Adam Stooke, Zhong Meng, CJ Zheng, Yanzhang He, Tara Sainath, Pedro Moreno Mengibar (2023)

## 🧩 Problem to Solve

본 논문은 음성 검색 쿼리와 같은 짧은 발화의 오프라인 인식을 위해 거대 규모(최대 20억 개의 파라미터)의 end-to-end 자동 음성 인식(ASR) 모델을 탐구한다. 특히, 최근 거대 언어 모델(LLM)의 발전과 더불어 ASR 모델의 크기를 확장했을 때 어떤 이점이 있는지, 그리고 서로 다른 두 가지 주요 모델 클래스인 Connectionist Temporal Classification (CTC)와 RNN-Transducer (RNN-T)가 대규모 설정에서 어떻게 비교되는지를 분석한다.

또한, 모델의 크기가 커질수록 Transformer나 Conformer와 같은 attention 기반 인코더의 연산 및 메모리 비용이 급격히 증가하는 문제가 발생한다. 특히 RNN-T는 prediction network와 joint network로 인한 추가 연산이 필요하여 학습 및 추론 효율성이 더욱 저하된다. 따라서 본 연구는 대규모 모델을 효율적으로 학습하고 추론하기 위한 방법론을 제시하고, 이를 통해 모델의 정확도와 효율성 사이의 최적의 균형점을 찾는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 대규모 ASR 모델에서 CTC와 RNN-T의 성능을 직접 비교하고, Funnel Pooling을 통한 공격적인 시간 축소(time reduction) 전략이 모델의 효율성과 정확도에 미치는 영향을 분석한 것이다.

중심적인 설계 아이디어는 인코더 내의 여러 레이어에 Funnel Pooling을 반복적으로 적용하여 프레임 속도(frame rate)를 획기적으로 낮추는 것이다. 이를 통해 연산량을 줄이면서도 대규모 모델의 성능을 유지할 수 있음을 보였다. 결과적으로 900M 파라미터의 RNN-T 모델이 1.8B 파라미터의 CTC 모델보다 더 높은 정확도를 보이며, 시간 축소에 대해서도 훨씬 더 강건(robust)하다는 사실을 밝혀냈다. 또한, 외부 언어 모델(LM)과의 shallow fusion이 CTC의 성능 격차를 상당히 줄일 수 있음을 입증하였다.

## 📎 Related Works

기존의 ASR 연구들은 Google의 Universal Speech Model (USM)과 같이 다국어 지원을 위해 모델 크기를 확장하는 경향이 있었으며, USM은 기본적으로 CTC 기반의 2B 파라미터 모델을 사용한다. 시간 축소(time reduction) 기법 역시 꾸준히 연구되어 왔으며, 초기에는 DNN이나 LSTM 모델에서 프레임을 연결(concatenating)하거나 계층적/피라미드형 RNN을 사용하는 방식이 사용되었다.

최근의 end-to-end 시스템에서는 합성곱 서브샘플링(convolutional subsampling)을 통해 40ms의 프레임 속도를 확보하는 것이 일반적이었으며, Conformer 아키텍처 기반의 음성 검색 연구에서는 초기 레이어에 2x 시간 축소를 적용해 60ms의 최종 프레임 속도를 구현한 사례가 있다. 본 논문은 이러한 기존 접근 방식보다 훨씬 더 공격적으로 시간 축소를 적용하여, 인코더 출력 단계에서 연산 비용을 대폭 절감하는 방향으로 차별화를 두었다.

## 🛠️ Methodology

### 전체 시스템 구조 및 인코더

본 연구는 Google의 USM 아키텍처를 따르는 Convolution-augmented Transformer인 Conformer를 인코더로 사용한다. 각 Conformer 블록은 Feed-forward module (FFN), Multi-head self-attention (MHSA), Convolution module, 그리고 두 번째 FFN으로 구성된다.

### 주요 모델 아키텍처

#### 1. Connectionist Temporal Classification (CTC)

CTC는 입력 시퀀스 $x$와 출력 시퀀스 $y$ 사이의 정렬(alignment)을 알 수 없을 때, 모든 가능한 유효 경로 $B_{ctc}(y)$에 대해 marginalization을 수행하여 조건부 분포 $P(y|x)$를 모델링한다.
$$P(y|x) = P(y|h(x)) = \sum_{a \in B_{ctc}(y)} \prod_{t=1}^{T} P(a_t|h)$$
CTC는 입력 특징이 주어졌을 때 출력 레이블들이 서로 독립적이라는 강한 조건부 독립 가정을 가진다.

#### 2. Neural Transducers (RNN-T)

RNN-T는 CTC의 독립성 가정을 완화하기 위해 prediction network를 도입하여 이전 예측 결과에 따라 다음 출력이 결정되도록 설계되었다.
$$P(y|x) = P(y|h(x)) = \sum_{a \in B_{nt}(y)} \prod_{\tau=1}^{T+U} P(a_\tau | q_{i_\tau}, h_{\tau-i_\tau})$$
여기서 $q_j$는 prediction network의 출력이며, 본 논문에서는 이전 두 개의 non-blank 레이블($y_{j-1}, y_{j-2}$)에만 의존하는 $\lvert V \rvert^2$ embedding network를 사용한다.

#### 3. Hybrid Autoregressive Transducer (HAT)

본 연구는 RNN-T의 변형인 HAT를 사용하여 출력 분포를 blank 심볼을 위한 Bernoulli 분포와 non-blank 레이블을 위한 분포로 분리한다. 또한, 모델 내부에서 학습된 internal language model (ILM)의 점수를 뺀 후 외부 LM ($P^{EXT}$)을 융합하는 shallow fusion 기법을 적용한다.
$$y^* = \arg \max_y \log P(y|x) - \alpha \log P^{ILM}(y) + \beta \log P^{EXT}(y)$$

### Funnel Pooling을 이용한 시간 축소

연산 효율성을 높이기 위해 MHSA 모듈 내에서 Strided-average pooling을 적용하는 Funnel Pooling 기법을 도입하였다. 쿼리(query) 벡터를 생성할 때만 풀링을 적용하고, 키(key)와 밸류(value) 시퀀스는 그대로 유지함으로써 정보 손실을 최소화하면서 시퀀스 길이를 줄인다.
$$\hat{x}^{(l)} = \text{StridedPooling}(\tilde{x}^{(l)})$$
$$x'^{(l)} = \hat{x}^{(l)} + \text{MHSA}(W_q \hat{x}^{(l)}, W_k \tilde{x}^{(l)}, W_v \tilde{x}^{(l)})$$
이 방식을 통해 인코더의 출력 프레임 속도를 40ms에서 최대 640ms까지 공격적으로 낮추어 학습 및 추론 속도를 개선하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: 490K 시간의 음성 검색 쿼리(약 5.2억 개 발화)를 학습에 사용하였다. 평가를 위해 실제 음성 데이터(VS-dev, VS-test)와 희귀 고유 명사(RPN)가 포함된 TTS 생성 데이터(RPNM, RPNN, RPNP, RPNS, RPNY)를 사용하였다.
- **지표**: Word Error Rate (WER)를 사용하였다.
- **비교 모델**: 340M 및 1.8B CTC 모델, 900M RNN-T 모델.

### 주요 결과

1. **CTC vs RNN-T 성능**: 동일한 프레임 속도(160ms)와 어휘 크기(16K)에서 900M RNN-T의 VS-dev WER은 3.7%로, 1.8B CTC의 4.2%보다 훨씬 우수하였다. 이는 레이블 의존성을 모델링할 수 있는 RNN-T의 구조적 이점을 보여준다.
2. **시간 축소에 대한 강건성**: RNN-T는 프레임 속도를 160ms에서 640ms까지 대폭 늘려도 VS-dev에서 WER 변화가 매우 적었다(3.7% $\rightarrow$ 3.9%). 반면 CTC는 320ms에서 성능이 급격히 저하(5.0%)되었다.
3. **LM Shallow Fusion**: 1.8B CTC 모델에 128M LM을 융합했을 때 VS-dev WER이 4.2%에서 3.8% 수준으로 낮아졌으며, 이는 RNN-T의 성능과 유사한 수준이다. 즉, LM 융합이 CTC의 독립성 가정으로 인한 손실을 보완할 수 있다.
4. **데이터 다양성의 영향**: RNN-T 모델을 다중 도메인(multi-domain) 데이터로 학습시켰을 때, 특히 뉴스 도메인(RPNN)과 같은 긴 발화에 대한 성능이 크게 향상되었으며, 시간 축소에 따른 성능 저하 폭도 줄어들었다.

## 🧠 Insights & Discussion

본 연구는 대규모 ASR 모델에서 모델의 크기보다 모델의 구조(특히 label dependency 모델링 여부)가 성능에 더 결정적인 영향을 미친다는 점을 시사한다. 1.8B의 거대 CTC 모델보다 900M의 상대적으로 작은 RNN-T 모델이 더 정확하다는 결과는, 단순한 파라미터 증량보다 적절한 아키텍처 설계가 중요함을 보여준다.

또한, Funnel Pooling을 통한 공격적인 시간 축소가 대규모 모델의 실용성을 높이는 핵심 요소임을 확인하였다. 다만, HAT 모델의 internal LM이 짧은 발화에 편향되어 학습될 경우, 긴 발화(RPNN 등)에서 shallow fusion 시 성능이 오히려 저하되는 현상이 관찰되었다. 이는 학습 데이터의 길이 다양성을 확보하는 것이 모델의 일반화 성능과 외부 LM 통합의 안정성에 필수적임을 의미한다.

비판적으로 분석하자면, 본 연구는 오프라인 인식에 초점을 맞추었기에 실시간 스트리밍 환경에서의 지연 시간(latency)과 Funnel Pooling의 상충 관계에 대해서는 충분히 다루지 않았다. 또한, CTC의 성능 저하가 특정 프레임 속도 임계값에서 발생하는 구체적인 이유에 대한 이론적 분석보다는 실험적 결과 제시 위주로 서술되어 있다.

## 📌 TL;DR

본 논문은 대규모 end-to-end ASR 모델에서 **RNN-T가 CTC보다 파라미터 효율성과 정확도 면에서 월등히 우수함**을 입증하였다. 특히 **Funnel Pooling**을 통해 프레임 속도를 획기적으로 낮추어도 RNN-T는 성능 저하 없이 효율적인 학습과 추론이 가능함을 보였으며, CTC의 성능 부족분은 외부 LM 융합으로 상당 부분 보완 가능함을 확인하였다. 이 연구는 향후 초거대 ASR 모델의 설계 방향이 단순한 크기 확장보다는 효율적인 시퀀스 압축과 레이블 의존성 모델링에 집중해야 함을 시사한다.
