# Enhancing Few-shot Keyword Spotting Performance through Pre-Trained Self-supervised Speech Models

Alican Gok, Oguzhan Buyuksolak, Osman Erman Okman, Murat Saraclar (2025)

## 🧩 Problem to Solve

본 논문은 배터리로 작동하는 엣지 디바이스(edge devices) 환경에서 사용자 정의 키워드를 적은 수의 샘플만으로 인식하게 하는 Few-Shot Keyword Spotting (FS-KWS)의 성능 향상을 목표로 한다.

전통적인 키워드 스포팅(KWS) 시스템은 고정된 어휘집에 대해서는 효과적이지만, 새로운 키워드를 추가하려면 수천 개의 학습 샘플이 필요하며 계산 및 메모리 자원 소모가 커서 저전력 임베디드 기기에 적용하기 어렵다는 한계가 있다. FS-KWS는 이를 해결하기 위해 소량의 데이터만으로 새로운 키워드를 등록하고 인식하려 하지만, 기존의 FS-KWS 시스템들은 특히 엣지 환경에서 낮은 오인식률(False Acceptance Rate, FAR)을 유지하면서 높은 정확도를 달성하는 데 어려움을 겪고 있다. 따라서 본 연구는 제한된 자원 환경에서도 강건하고 정확한 FS-KWS를 구현하기 위한 효율적인 학습 체계를 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사전 학습된 자기지도학습(Self-supervised Learning, SSL) 모델의 풍부한 표현력을 활용하여, 이를 엣지 디바이스에 적합한 경량 모델로 전이시키는 Teacher-Student 구조의 학습 프레임워크를 구축하는 것이다.

주요 기여 사항은 다음과 같다.

1. **SCAF (Sub-center ArcFace) 손실 함수의 도입**: 오디오 판별 및 FS-KWS 분야에 SCAF를 최초로 적용하여, 클래스 간 분리도(inter-class separability)를 높이고 클래스 내 응집도(intra-class compactness)를 강화함으로써 고차원 임베딩의 차원을 효율적으로 축소했다.
2. **Teacher-Student 지식 증류(Knowledge Distillation) 구조**: 고성능 SSL 모델 기반의 Teacher 모델이 생성한 최적의 임베딩을 경량화된 ResNet-15 Student 모델이 학습하도록 하여 엣지 디바이스에서의 성능을 극대화했다.
3. **효율적인 차원 축소(DR) 모델 제안**: Attention 기반의 차원 축소 구조를 통해 SSL 모델의 고차원 특징에서 핵심적인 시간적 정보를 보존하며 64차원의 컴팩트한 임베딩을 생성하는 방법을 제시했다.

## 📎 Related Works

기존의 FS-KWS는 주로 Metric Learning 프레임워크를 사용하며, 특히 Prototypical Networks가 널리 사용되었다. 이는 각 키워드 클래스의 임베딩 평균값인 프로토타입(prototype)을 생성하고, 테스트 샘플과의 거리를 측정하여 분류하는 방식이다. 주로 Triplet Loss나 Prototypical Loss가 사용되었으나, 낮은 FAR 환경에서 사용자 경험을 만족시킬 만한 정확도를 제공하지 못하는 한계가 있었다.

또한, 음성 작업의 성능을 높이기 위해 task-agnostic한 SSL 모델을 통합하는 시도가 있었다. 기존 연구들은 단순 평균 풀링(mean-pooling) 후 선형 변환을 사용하거나, 2층 컨볼루션 인코더를 사용하는 방식을 취했다. 본 논문은 이러한 기존 방식과 달리, Attention 기반의 DR 모델과 SCAF 손실 함수를 결합하여 더 강력한 판별력을 가진 임베딩 공간을 학습시킨다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Teacher Representation Model

Teacher 모델은 Wav2Vec 2.0의 16번째 트랜스포머 레이어에서 추출된 고차원 임베딩을 입력으로 받는다. 1초 길이의 음성 데이터는 $49 \times 1024$ 차원의 특징으로 표현되는데, 이를 엣지 디바이스에서 처리 가능한 64차원으로 축소하기 위해 두 가지 DR(Dimensionality Reduction) 구조를 검토하였다.

- **Linear Projection**: 시간 축에 대해 평균 풀링(average pooling)을 수행한 후 선형 투영을 통해 차원을 축소한다.
- **Attention Encoder**: Scaled Dot Product Attention 메커니즘을 통해 시간적 관계를 계산하고, 이후 1D Convolution 레이어를 통해 가중 평균을 수행함으로써 중요한 시간적 특징을 보존하며 차원을 축소한다.

학습 시에는 **Sub-center ArcFace (SCAF)** 손실 함수를 사용한다. SCAF는 클래스당 하나의 중심이 아닌 여러 개의 서브 센터(sub-centers)를 두어 클래스 내의 변동성을 더 효과적으로 수용하며, 각 샘플을 가장 가까운 서브 센터에 할당하여 클래스 간 거리와 클래스 내 응집력을 최적화한다.

### 2. Edge-friendly Student Representation Model

Student 모델은 메모리 및 계산 효율성을 위해 **ResNet-15** 아키텍처를 사용하며, 입력값으로는 오디오에서 추출한 10개의 MFCC(Mel-frequency cepstral coefficients) 특징 맵($49 \times 10$)을 사용한다.

### 3. Knowledge Distillation (KD) 및 학습 절차

Teacher 모델의 지식을 Student 모델로 전이하기 위해 지식 증류 기법을 적용한다. Student 모델의 최종 손실 함수 $L$은 다음과 같이 정의된다.

$$L = L_{KD} + \lambda L_T$$

여기서 각 항의 의미는 다음과 같다.

- $L_{KD}$: 지식 증류 손실로, Teacher 모델과 Student 모델이 생성한 임베딩 간의 차이를 줄이기 위해 **Mean Squared Error (MSE)**를 사용한다.
- $L_T$: 특정 작업 수행을 위한 Task-specific loss로, Triplet loss 또는 SCAF loss가 사용된다.
- $\lambda$: 두 손실 함수의 균형을 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: MSWC(Multilingual Spoken Words Corpus) 영어 부분 및 Google Speech Commands (GSC) 데이터셋을 사용하였다.
- **평가 지표**: 1-shot, 5-shot, 10-shot 설정에서 정확도(Accuracy), 오인식률(FAR), AUC, DET(Detection Error Trade-off) 등을 측정하였다.
- **추론 방식**: 등록 단계에서 $K$개의 샘플로 프로토타입(평균 임베딩)을 생성하고, 테스트 단계에서 코사인 거리(cosine distance)를 측정하여 임계값 $T$보다 작으면 해당 클래스로 분류하고, 그렇지 않으면 "others" 클래스로 분류한다.

### 주요 결과

- **DR 모델 선택**: Attention 기반 DR 모델이 Linear Projection보다 월등한 성능을 보였으며, 특히 SCAF 손실 함수와 결합했을 때 가장 높은 성능을 나타냈다.
- **GSC 데이터셋 성능 (10-shot)**: 제안된 KD 기반 학습 방법은 1% FAR 기준에서 분류 정확도를 기존 baseline(Triplet loss 사용)의 **33.4%에서 74.1%로 대폭 향상**시켰다.
- **MSWC 데이터셋 성능**: $L_{KD}$와 Triplet loss를 결합한 방식($KD + Triplet$)이 일부 지표에서 우수했으나, 전반적으로 KD 단독 사용 또는 결합 방식이 baseline보다 높은 성능을 보였다.

| 모델 및 전략 | GSC Accuracy (10-shot, 1% FAR) | GSC AUC | MSWC AUROC (10-shot) |
| :--- | :---: | :---: | :---: |
| Baseline (Triplet) | 33.4% | 91.4% | 99.4% |
| SCAF (Student only) | 47.5% | 91.0% | 97.4% |
| **KD (Proposed)** | **74.1%** | **95.5%** | **99.6%** |
| KD + Triplet | 56.5% | 94.8% | 99.7% |
| KD + SCAF | 63.7% | 91.6% | 99.1% |

## 🧠 Insights & Discussion

본 연구는 SSL 모델의 강력한 특징 추출 능력을 엣지 모델에 효율적으로 전이시키는 파이프라인을 성공적으로 구축하였다. 특히, GSC 데이터셋과 같은 **Cross-domain 설정**(학습 데이터와 테스트 데이터의 화자나 녹음 환경이 다른 경우)에서는 Task-specific loss($L_T$)를 제외하고 **KD 손실만 사용했을 때 가장 좋은 성능**이 나타났다. 이는 특정 작업에 과적합(overfitting)되는 것을 방지하고 SSL 모델이 가진 일반적인 음성 표현력을 그대로 가져오는 것이 강건성(robustness) 확보에 더 유리함을 시사한다.

반면, MSWC 데이터셋처럼 동일 도메인 내에서 테스트할 때는 $L_{KD}$와 $L_T$를 결합하는 것이 성능을 소폭 향상시켰는데, 이는 이미 확보된 일반적 특징에 해당 데이터셋의 특성을 추가로 학습시켰기 때문으로 해석된다. 결과적으로, 실제 환경에서의 범용적인 적용을 위해서는 KD 기반의 학습 전략이 가장 권장된다.

## 📌 TL;DR

본 논문은 사전 학습된 Wav2Vec 2.0 SSL 모델과 SCAF 손실 함수를 이용한 Teacher 모델을 구축하고, 지식 증류(KD)를 통해 경량 ResNet-15 Student 모델을 학습시켜 엣지 디바이스용 Few-shot Keyword Spotting 성능을 높였다. 특히 GSC 데이터셋의 10-shot 설정에서 1% FAR 기준 정확도를 33.4%에서 74.1%로 비약적으로 향상시켰으며, 이는 SSL 모델의 범용적 특징을 경량 모델에 이식하는 것이 엣지 환경의 KWS 성능 향상에 핵심적임을 입증한다.
