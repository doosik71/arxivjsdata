# I2CR: Improving Noise Robustness on Keyword Spotting using Inter-Intra Contrastive Regularization

Dianwen Ng, Jia Qi Yip, Tanmay Surana, Zhao Yang, Chong Zhang, Yukun Ma, Chongjia Ni, Eng Siong Chng, and Bin Ma (2022)

## 🧩 Problem to Solve

본 논문은 키워드 검출(Keyword Spotting, KWS) 시스템이 소음이 심한 환경, 특히 신호 대 잡음비(Signal-to-Noise Ratio, SNR)가 낮은 환경에서 성능이 급격히 저하되는 문제를 해결하고자 한다.

KWS 시스템은 Siri나 Google Assistant와 같은 개인 비서 서비스의 핵심 기술이지만, 대부분의 기존 모델은 깨끗한 오디오나 근접 거리의 음성 데이터에 최적화되어 있다. 소음 환경에 대응하기 위해 다양한 SNR 수준의 소음을 추가하여 학습시키는 Multi-conditioning training 방식이 사용되기도 하지만, 이는 모델이 특정 SNR 범위에 과적합될 가능성이 크며, 학습 시 경험하지 못한 새로운 종류의 소음(Out-of-domain noise)이나 매우 낮은 SNR 환경에서는 여전히 취약하다는 한계가 있다. 따라서 본 연구의 목표는 모델이 소음과 같은 불필요한 특징(Nuisance features)을 억제하고, 클래스 고유의 핵심적인 음성 정보만을 추출하여 소음 강건성(Noise Robustness)을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Inter-Intra Contrastive Regularization (I2CR)**이라는 대조 학습 기반의 정규화 방법을 도입하는 것이다.

기존의 대조 학습이 주로 한 샘플의 서로 다른 증강 뷰(Augmented view)만을 긍정 쌍(Positive pair)으로 사용하는 것과 달리, I2CR은 다음 두 가지를 모두 긍정 쌍으로 활용한다:

1. **Intra-view**: 동일한 샘플에 서로 다른 소음 증강을 적용한 경우.
2. **Inter-view**: 동일한 클래스에 속하지만 서로 다른 화자가 말한 다른 샘플인 경우.

이러한 설계를 통해 모델은 특정 화자의 스타일이나 특정 소음 패턴에 의존하지 않고, 해당 키워드 클래스가 공통적으로 가지는 일반화된 특징(Generalized representations)을 학습하게 된다. 결과적으로 임베딩 공간에서 동일 클래스 샘플들이 더 조밀하고 명확한 클러스터를 형성하게 되어 소음의 영향을 덜 받게 된다.

## 📎 Related Works

기존의 KWS 연구들은 주로 모델 아키텍처의 효율성을 높이거나 깨끗한 데이터셋에서의 정확도를 올리는 데 집중하였다. 소음 강건성을 위해 일부 연구에서는 self-supervised contrastive learning을 이용한 사전 학습(Pre-training) 후 미세 조정(Fine-tuning)하는 2단계 전략을 사용하였다. 하지만 이러한 방식은 대규모의 외부 데이터셋이 필요하며 학습 리소스 소모가 크다는 단점이 있다.

또한, 이미지 분류 분야에서는 대조 손실(Contrastive loss)을 정규화 도구(Regularizer)로 사용하여 도메인 일반화(Domain generalization) 성능을 높인 사례가 있다. 본 논문은 이러한 아이디어를 KWS 분야에 접목하되, 단순한 샘플 내 증강을 넘어 클래스 내 서로 다른 샘플 간의 관계까지 고려하는 I2CR 방식을 제안함으로써 기존의 단순 대조 학습이나 교차 엔트로피(Cross Entropy) 기반 학습보다 더 강력한 강건성을 확보하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 제안 방법은 기존의 지도 학습(Supervised Learning) 구조에 I2CR 정규화 항을 추가한 형태이다. 모델 아키텍처의 변경 없이 학습 목적 함수(Objective function)만을 수정하여 적용할 수 있다.

### 2. 학습 목적 함수 및 손실 함수

전체 손실 함수 $L$은 지도 학습을 위한 교차 엔트로피 손실($L_{CE}$)과 I2CR 정규화 손실($L_{I2CR}$)의 가중 합으로 정의된다.

$$L = L_{CE} + \alpha L_{I2CR}$$

여기서 $\alpha$는 정규화의 강도를 조절하는 하이퍼파라미터이다. I2CR 손실 함수는 다음과 같이 정의된다:

$$L_{I2CR} = \sum_{i \in I} -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{n \notin P(i)} \exp(\text{sim}(z_i, z_n)/\tau)}$$

- $z_i, z_p, z_n$: 각각 앵커(Anchor), 긍정(Positive), 부정(Negative) 샘플의 정규화된 임베딩 벡터이다.
- $P(i)$: 샘플 $i$와 동일한 클래스에 속하는 모든 샘플(자기 자신의 증강 뷰 포함)의 인덱스 집합이다.
- $\text{sim}(z_i, z_p)$: 두 벡터 간의 코사인 유사도(Cosine similarity)를 의미한다.
- $\tau$: 온도 파라미터(Temperature parameter)로, 분포의 Sharpness를 조절한다.

### 3. 데이터 증강(Data Augmentation) 파이프라인

효과적인 대조 학습을 위해 다음과 같은 증강 기법을 적용한다:

- **Noise Augmentation**: FSD50K 데이터셋에서 무작위 소음을 선택하여 $-10\text{dB}$에서 $30\text{dB}$ 사이의 SNR로 추가한다.
- **Time-frequency Masking**: 스펙트로그램의 일부 시간축과 주파수축 영역을 0으로 마스킹한다.
- **Time Shifting**: 오디오 샘플을 $\pm 100\text{ms}$ 범위 내에서 무작위로 이동시킨다.
- **Speed Perturbation**: 오디오의 속도를 $90\% \sim 110\%$ 범위에서 무작위로 변경한다.

### 4. 학습 절차 및 전략

학습 초기에는 가중치들이 무작위로 초기화되어 있어, 서로 다른 샘플 간의 유사도를 강제로 높이는 것이 학습을 불안정하게 만들 수 있다. 이를 방지하기 위해 $\alpha$ 값을 다음과 같이 스케줄링한다:

- 첫 번째 에포크(Epoch)에서는 $\alpha = 0$으로 설정하여 $L_{CE}$로만 학습한다.
- 이후 전체 에포크 수에 따라 $\alpha$를 선형적으로 증가시키며, 최대값은 $0.5$로 제한한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Google Speech Commands V2 (10개 클래스 및 35개 클래스 서브셋).
- **백본 모델**: EfficientNet-B0, ResNet-18, Keyword Transformer (KWT-1).
- **평가 지표**: 다양한 소음 유형(Traffic, Metro, Car) 및 SNR 수준($-10, 0, 20\text{dB}$)에서의 정확도(Accuracy).
- **비교 대상**: Baseline (Vanilla Supervised), Intra-view Contrastive Regularization (샘플 내 증강만 사용).

### 2. 주요 결과

- **정량적 성과**: 모든 백본 모델에서 I2CR을 적용했을 때 정확도가 향상되었다. 특히 SNR이 낮은 $-10\text{dB}$ 환경에서 성능 향상 폭이 가장 컸으며, KWT-1 모델의 경우 Car noise 환경에서 최대 $4\%$의 절대적 정확도 향상을 보였다.
- **강건성 검증 (Out-of-domain & Extreme SNR)**:
  - **Extreme SNR**: 학습 시 경험하지 않은 $-30, -20, -15\text{dB}$ 환경에서도 I2CR 모델이 Baseline보다 월등한 성능을 보였다. (10개 클래스 기준 $-30\text{dB}$에서 평균 $4\%$ 향상)
  - **Out-of-domain Noise**: 학습에 사용되지 않은 Google Speech Commands의 배경 소음을 적용했을 때, $-10\text{dB}$에서 10개 클래스 모델 기준 평균 $5.7\%$의 정확도 향상을 기록했다.
- **정성적 분석**: t-SNE 시각화 결과, I2CR을 적용한 모델의 임베딩 공간에서 클래스별 클러스터가 Baseline보다 훨씬 더 명확하고 조밀하게 형성됨을 확인하였다. 이는 모델이 소음의 영향을 덜 받고 클래스 고유의 특징을 잘 학습했음을 시사한다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 모델 아키텍처의 변경 없이 학습 목적 함수를 수정하는 것만으로도 소음 강건성을 획기적으로 개선했다는 점이다. 특히 **Inter-view** contrasting을 도입함으로써, 동일 샘플의 증강 뷰만으로는 해결할 수 없었던 '화자 고유의 특성'이나 '특정 소음 패턴'으로 인한 편향(Bias)을 제거하고, 클래스 전체를 아우르는 일반적인 특징을 학습하게 만들었다.

실험 결과에서 나타난 unseen noise 및 extreme SNR에 대한 성능 향상은 I2CR이 단순히 학습 데이터에 맞춘 것이 아니라, 진정한 의미의 **표현 학습(Representation Learning)**을 달성했음을 보여준다.

다만, 본 논문에서는 $\alpha$의 스케줄링이나 증강 기법의 조합이 성능에 미치는 영향에 대해 상세한 ablation study를 제공하지 않은 점이 아쉬움으로 남는다. 또한, 실시간 KWS 시스템에서 추론 속도에는 영향이 없으나, 학습 단계에서 긍정 쌍을 구성하기 위한 메모리 및 연산 비용 증가분량에 대한 분석이 추가되었다면 더 완벽했을 것이다.

## 📌 TL;DR

본 논문은 KWS 모델의 소음 강건성을 높이기 위해 **Inter-Intra Contrastive Regularization (I2CR)**을 제안한다. I2CR은 동일 샘플의 증강 뷰(Intra)와 동일 클래스의 타 샘플(Inter)을 모두 긍정 쌍으로 활용하여 소음과 무관한 일반화된 클래스 특징을 학습하도록 유도한다. 실험 결과, 다양한 백본 모델에서 성능 향상이 확인되었으며, 특히 학습하지 않은 소음 환경이나 극심한 저SNR 환경에서도 뛰어난 일반화 성능을 보여 실제 소음이 심한 환경의 KWS 시스템 적용 가능성을 입증하였다.
