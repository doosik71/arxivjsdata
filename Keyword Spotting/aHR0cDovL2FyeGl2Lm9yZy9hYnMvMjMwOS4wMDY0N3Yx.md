# Improving Small Footprint Few-shot Keyword Spotting with Supervision on Auxiliary Data

Seunghan Yang, Byeonggeun Kim, Kyuhong Shim, Simyung Chang (2023)

## 🧩 Problem to Solve

본 논문은 적은 수의 샘플만으로 새로운 키워드를 인식해야 하는 Few-shot Keyword Spotting (FS-KWS) 문제에서 모델의 일반화 성능을 높이기 위한 데이터 부족 문제를 해결하고자 한다.

일반적으로 FS-KWS 모델이 보지 못한(unseen) 타겟 키워드에 대해 잘 일반화하기 위해서는 대규모의 어노테이션된(annotated) 데이터셋이 필요하다. 하지만 기존의 KWS 데이터셋은 규모가 제한적이며, 키워드 형태의 라벨링된 데이터를 추가로 수집하는 것은 비용이 매우 많이 드는 작업이다. 

따라서 본 연구의 목표는 쉽게 수집 가능한 대규모의 비라벨링 낭독 음성 데이터(reading speech data)를 보조 데이터(auxiliary data)로 활용하여, 적은 파라미터 수를 가진 Small Footprint 모델에서도 효율적으로 동작하는 FS-KWS 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대규모 비라벨링 데이터를 단순히 Self-supervised Learning (SSL)으로 사용하는 대신, 이를 자동으로 라벨링하고 필터링하여 지도 학습(Supervised Learning)이 가능한 형태로 변환하여 사용하는 것이다.

1.  **LibriWord 데이터셋 구축**: LibriSpeech와 같은 낭독 음성 데이터에서 단어 추출 기술을 통해 단어 수준의 라벨을 자동으로 생성하고, 데이터 불균형을 해소하여 구축한 키워드 유사 데이터셋이다.
2.  **AuxSL 프레임워크 제안**: In-domain 데이터(명령어 형태)와 Out-of-domain 보조 데이터(낭독 형태) 간의 도메인 간극(domain gap)으로 인한 표현력 왜곡을 방지하기 위해, 보조 데이터 전용 분류기를 추가한 Multi-task Learning (MTL) 구조를 제안한다.

## 📎 Related Works

### 기존 연구 및 한계
- **Conventional KWS**: 정해진 소수의 키워드(예: "Alexa", "OK Google")를 감지하는 데 집중하며, 메모리 및 전력 소비 최소화에 초점을 맞춘다.
- **Few-shot KWS**: 사용자 정의 키워드를 지원하기 위해 Metric Learning 기반의 방법론들이 제안되었다. 주로 대규모 학습 데이터셋을 통해 강건한 임베딩 공간을 학습하는 방식이다.
- **SSL 기반 접근**: 라벨 없는 데이터에서 표현력을 학습하기 위해 SimCLR, BYOL, wav2vec 2.0 등이 사용된다. 그러나 이러한 SSL 방법론은 모델의 용량(capacity)이 충분히 큰 대형 모델에서만 효과적이며, KWS와 같은 Small Footprint 모델에서는 실용적이지 않다는 한계가 있다.

### 차별점
본 논문은 SSL의 한계를 지적하며, 보조 데이터를 지도 학습 가능하게 가공(LibriWord)하고 이를 MTL 구조(AuxSL)로 학습시킴으로써 소형 모델에서도 성능 향상을 이끌어냈다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. Problem Definition: Metric Learning
FS-KWS는 거리 측정 방식을 통해 새로운 클래스를 분류한다. 대표적으로 ProtoNets는 각 클래스의 임베딩 평균인 프로토타입 $c_n$을 생성한다.

$$c_n = \frac{1}{K} \sum_{i=1}^{K} F_{\theta}(x_{s_{n,i}})$$

여기서 $F_{\theta}(\cdot)$는 특징 추출기(feature extractor)이다. 쿼리 샘플 $x_{q_{n,i}}$에 대한 손실 함수 $L_{FSL}$은 다음과 같이 정의된다.

$$L_{FSL} = -\frac{1}{M} \sum_{i=1}^{M} \log p_{\theta}(y=n|x_{q_{n,i}})$$

$$p_{\theta}(y=n|x_{q_{n,i}}) = \frac{\exp(-d(F_{\theta}(x_{q_{n,i}}), c_n))}{\sum_{n'=1}^{N} \exp(-d(F_{\theta}(x_{q_{n,i}}), c_{n'}))}$$

### 2. LibriWord 데이터셋 구축
비라벨링 데이터인 LibriSpeech 코퍼스를 활용하여 다음과 같은 절차로 LibriWord를 생성한다.
- **단어 추출**: Montreal Forced Aligner를 사용하여 음성과 텍스트 간의 정렬을 통해 단어 수준의 세그먼트를 추출한다.
- **필터링 및 밸런싱**: 과거형, 복수형 등 유사한 단어 중 하나만 남기고 제거하여 중복성을 줄이며, 상위 1,000개 빈도 키워드에 대해 각 300개씩 샘플을 할당하여 균형 잡힌 데이터셋을 구성한다.

### 3. AuxSL (Auxiliary Supervision Learning) 프레임워크
In-domain 데이터와 Out-of-domain 보조 데이터를 동시에 학습할 때 발생하는 표현력 왜곡을 방지하기 위해, 보조 데이터만을 위한 별도의 분류기 $C_{\phi}(\cdot)$를 둔 MTL 구조를 사용한다.

전체 학습 목표 함수는 다음과 같다.
$$L_{AuxSL} = L_{FSL} + \lambda L_{SL}$$

- $L_{FSL}$: In-domain 데이터에 적용되는 Few-shot Learning 손실 함수 (본 논문에서는 D-ProtoNets의 Dummy prototypical loss 사용).
- $L_{SL}$: Out-of-domain 보조 데이터에 적용되는 일반적인 Cross-Entropy 손실 함수.
- $\lambda$: 두 손실 함수의 균형을 맞추는 하이퍼파라미터.

추론 단계에서는 보조 데이터용 분류기 $C_{\phi}(\cdot)$를 사용하지 않으므로 추가적인 연산 비용이 발생하지 않는다.

## 📊 Results

### 실험 설정
- **데이터셋**: Google Speech Commands (GSC)의 splitGSC 벤치마크 사용.
- **백본 모델**: BC-ResNet8 (321k), Res12 (8M), DS-ResNet18 (72k).
- **평가 지표**: 1-shot 및 5-shot 설정에서 Closed-set Accuracy (%)와 Open-set AUROC를 측정.

### 주요 결과
1.  **데이터 구성의 영향**: 보조 데이터를 불균형하게 사용하거나 전체 LibriSpeech를 사용하는 것보다, 균형 있게 정제된 LibriWord를 사용할 때 성능이 더 높게 나타났다 (Figure 3).
2.  **SSL vs SL 비교**: 
    - 소형 모델에서 SSL(SimCLR, BYOL 등) 기반의 사전 학습이나 MTL은 성능 향상이 미비하거나 오히려 성능을 저하시켰다.
    - 반면, 보조 데이터를 지도 학습으로 활용한 AuxSL은 Baseline(D-Proto) 대비 5-shot Accuracy에서 약 16%의 상대적 향상을 보였다 (Table 2).
3.  **모델 크기별 성능**: 
    - 모델 크기가 커질수록 모든 방법론의 성능이 향상되지만, AuxSL의 효율성이 압도적이다.
    - 특히, AuxSL을 적용한 가장 작은 모델(BC-ResNet1, 9.2k)이 AuxSL을 적용하지 않은 가장 큰 모델(BC-ResNet8, 321k)보다 더 높은 성능을 기록하였다 (Figure 4).

## 🧠 Insights & Discussion

### 강점
본 논문은 대규모 데이터를 활용하는 일반적인 방법인 SSL이 소형 모델에서는 부적합하다는 점을 실험적으로 증명하였다. 대신, 데이터를 정교하게 정제하여 지도 학습 형태로 제공하는 것이 소형 모델의 임베딩 공간을 형성하는 데 훨씬 효과적임을 보여주었다. 또한, MTL 구조를 통해 도메인 간극을 효과적으로 처리하여 모델의 일반화 능력을 극대화하였다.

### 한계 및 향후 과제
- **도메인 간극**: 낭독 음성과 실제 명령어 음성 사이의 도메인 차이가 여전히 존재한다. 
- **미해결 질문**: 본 논문에서는 단순한 MTL 구조를 제안했으나, 향후 연구에서 RFN (Relaxed Instance Frequency-wise Normalization)이나 DSBN (Domain-Specific Batch Normalization)과 같은 더 정교한 도메인 일반화 기법을 적용했을 때의 효과를 분석할 필요가 있다.

## 📌 TL;DR

본 논문은 소형 모델 기반의 Few-shot Keyword Spotting 성능을 높이기 위해, 대규모 낭독 음성 데이터를 정제하여 구축한 **LibriWord 데이터셋**과 이를 효율적으로 학습시키기 위한 **AuxSL(Multi-task Learning) 프레임워크**를 제안한다. 실험 결과, 소형 모델에서는 SSL보다 정제된 보조 데이터를 이용한 지도 학습이 훨씬 효과적이며, 매우 작은 모델로도 대형 모델 이상의 성능을 낼 수 있음을 입증하였다. 이 연구는 온디바이스(on-device) 환경의 저전력/저사양 KWS 시스템 구현에 중요한 기여를 할 것으로 보인다.