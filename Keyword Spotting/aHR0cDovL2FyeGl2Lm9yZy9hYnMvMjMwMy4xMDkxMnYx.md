# EXPLORING REPRESENTATION LEARNING FOR SMALL-FOOTPRINT KEYWORD SPOTTING

Fan Cui, Liyong Guo, Quandong Wang, Peng Gao, Yujun Wang (2023)

## 🧩 Problem to Solve

본 논문은 저리소스 환경에서의 키워드 검출(Keyword Spotting, KWS) 시스템이 직면한 두 가지 핵심 문제를 해결하고자 한다. 첫 번째는 학습에 사용할 수 있는 레이블링된 데이터(labeled data)가 매우 제한적이라는 점이며, 두 번째는 KWS 시스템이 주로 임베디드 기기나 모바일 기기에 탑재되어야 하므로 가용한 컴퓨팅 자원(메모리 및 연산량)이 극히 제한적이라는 점이다.

기존의 딥러닝 기반 KWS 방법론, 특히 CNN 기반 모델들은 높은 정확도를 보이지만 대량의 레이블링된 데이터를 필요로 하며, 녹음 환경의 변화로 인한 데이터 분포의 불일치(data distribution shifting) 문제가 발생할 때 모델 성능이 저하되는 한계가 있다. 따라서 본 연구의 목표는 적은 양의 레이블링된 데이터만으로도 높은 성능을 낼 수 있도록, 자기지도 학습(self-supervised learning) 기반의 표현 학습(representation learning)을 통해 소형 모델(small-footprint model)의 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대규모 unlabeled 데이터를 활용하여 모델이 의미 있는 음향 표현을 먼저 학습하게 한 뒤, 이를 소형 KWS 모델에 전이하는 것이다. 이를 위해 다음과 같은 세 가지 설계를 제안한다.

1. **LGCSiam (Local-Global Contrastive Siamese networks):** 오디오 샘플의 로컬(프레임 수준)과 글로벌(발화 수준) 특성을 모두 학습하는 대조 학습 구조를 설계하여, 정답 레이블 없이도 유사한 오디오 샘플이 유사한 표현을 갖도록 유도한다.
2. **WVC (Wav2Vec 2.0 Constraint) 모듈:** 대규모 데이터로 사전 학습된 Wav2Vec 2.0 모델을 교사(Teacher) 모델로 활용하여, 소형 KWS 모델의 인코더가 고차원의 풍부한 음향 표현을 학습하도록 강제하는 제약 조건을 부여한다.
3. **TCANet 아키텍처:** 연산 효율성을 위해 Temporal Convolution Network 기반의 인코더와 Multi-head Self-Attention 기반의 디코더를 결합한 경량 모델을 제안한다.

## 📎 Related Works

논문에서는 기존의 KWS 접근 방식과 사전 학습 모델의 활용 가능성을 언급한다.

- **기존 KWS 모델:** DS-CNN이나 TC-ResNet과 같은 CNN 기반 모델들이 효율성과 정확도 면에서 우수한 성과를 거두었으나, 여전히 대량의 레이블링된 데이터에 의존한다는 한계가 있다.
- **사전 학습 모델 (Self-supervised Pretrained Models):** NLP의 BERT나 음성 인식의 Wav2Vec 2.0과 같이 대량의 unlabeled 데이터로 학습된 모델들이 다운스트림 태스크에서 뛰어난 성능을 보임을 확인하였다. 
- **차별점:** 기존의 사전 학습 모델을 활용한 KWS 연구는 모델의 크기가 너무 커서 저리소스 기기에 배포하기 어렵다는 단점이 있었다. 본 논문은 거대 모델을 직접 사용하는 대신, 거대 모델의 표현력을 소형 모델(TCANet)에 전이시키는 방식(Teacher-Student learning 및 Contrastive learning)을 채택하여 배포 가능성과 성능을 동시에 확보했다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. TCANet Model Architecture
TCANet은 입력 파형을 Log-Mel Spectrogram($x \in \mathbb{R}^{B \times T \times 40}$)으로 변환하여 처리하며, 크게 인코더와 디코더로 구성된다.

- **Encoder $E(\cdot)$:** 7개의 합성곱 계층(Convolutional layers)으로 구성된다. 파라미터 수를 줄이기 위해 첫 번째 계층을 제외하고는 모두 Separable Conv1d를 사용한다. 인코더를 거치면 입력 데이터는 $e \in \mathbb{R}^{B \times T/2 \times 64}$ 형태로 변환된다.
- **Decoder $D(\cdot)$:** Multi-head Attention 블록을 사용하여 시간적 의존성을 학습한다. Query($Q$), Key($K$), Value($V$)는 다음과 같이 선형 투영을 통해 얻는다.
  $$Q=e \cdot W^Q, \quad K=e \cdot W^K, \quad V=e \cdot W^V$$
  각 헤드의 출력은 $\text{Softmax}(\frac{Q_i K_i^T}{\sqrt{n}})V_i$로 계산되며, 최종 결과는 모든 헤드를 결합(Concat)한 후 $W^o$ 행렬을 통해 투영되어 $D(e)$가 된다.
- **Classification:** 지도 학습 시에는 Global Average Pooling을 통해 시간축을 통합하고, Dense 및 Softmax 계층을 통해 클래스 확률 $p$를 출력한다. 손실 함수로는 Cross Entropy (CE) loss를 사용한다.
  $$\text{Loss}_{CE} = -\sum_{i=1}^{B} \sum_{j=1}^{C} y_{ij} \log(p_{ij})$$

### 2. WVC Module
사전 학습된 Wav2Vec 2.0 모델의 지식을 전이하기 위한 모듈이다.
- **작동 원리:** Wav2Vec 2.0 모델에서 추출한 컨텍스트 표현 $e_0 \in \mathbb{R}^{B \times T/2 \times 768}$와 TCANet 인코더의 출력 $e_1 \in \mathbb{R}^{B \times T/2 \times 64}$ 사이의 간극을 줄인다.
- **Projection Head:** MLP 블록을 사용하여 $e_1$을 $e'_1$으로 변환하여 차원을 맞춘다.
  $$e'_1 = W_2 \sigma(W_1 \cdot e_1)$$
- **Loss:** 두 표현의 유사도를 MSE(Mean Squared Error) loss로 측정하여 인코더가 풍부한 음향 표현을 학습하게 한다.
  $$\text{Loss}_{WVC} = \| e'_1 - e_0 \|^2$$

### 3. LGCSiam Module
레이블이 없는 데이터에서 발화의 특성을 학습하기 위한 Siamese 네트워크 구조이다.
- **구조:** 동일한 가중치를 공유하는 두 개의 브랜치에 서로 다른 데이터 증강(augmentation)이 적용된 동일 샘플 $x_1, x_2$를 입력한다.
- **Loss 구성:**
    - **Global Consistency Loss ($L_{global}$):** 두 샘플의 전체 시간축 평균 표현이 서로 가까워지도록 유도한다.
      $$L_{global} = \frac{1}{B} \sum_{j=1}^{B} \{ [ \frac{2}{T} \sum_{i=0}^{T/2} d'_1(j,i,:) - \frac{2}{T} \sum_{i=0}^{T/2} d'_2(j,i,:) ]^2 \}$$
    - **Local Contrastive Loss ($L_{local}$):** 동일 샘플의 동일 타임스탬프 프레임은 가깝게, 다른 샘플이나 다른 타임스탬프의 프레임은 멀게 학습시킨다. 코사인 유사도($\text{sim}$)를 기반으로 InfoNCE 스타일의 손실 함수를 사용한다.
      $$L_{local} = -\log \frac{\exp(\text{sim}(d'_1(i_1,j_1), d'_2(i_1,j_1))/\tau)}{\sum \exp(\text{sim}(d'_1(i_1,j_1), d'_2(i_2,j_2))/\tau)}$$
- **최종 손실 함수:** $\text{Loss}_{LGCSiam} = L_{global} + L_{local}$

### 4. 학습 절차
학습은 크게 두 단계로 나뉜다.
1. **Pre-training Stage:** 
   - 먼저 $\text{Loss}_{WVC}$로 인코더를 최적화한다.
   - 이후 $\text{Loss} = \lambda_1 \text{Loss}_{LGCSiam} + \lambda_2 \text{Loss}_{WVC}$를 통해 인코더와 디코더를 동시에 학습시킨다.
2. **Supervised Fine-tuning Stage:** 
   - 분류 모듈을 추가하고 다음과 같은 통합 손실 함수를 사용하여 미세 조정한다.
     $$\text{Loss} = \gamma_1 \text{Loss}_{CE} + \gamma_2 \text{Loss}_{LGCSiam} + \gamma_3 \text{Loss}_{WVC}$$

## 📊 Results

### 1. 실험 설정
- **데이터셋:** Speech Commands (SC) 데이터셋 (12개 클래스). 교차 도메인 실험을 위해 unlabeled AISHELL 데이터셋을 사전 학습에 사용하였다.
- **데이터 증강:** Pre-emphasize, Pitch shift, Notch/Peak filter, Background noise 추가, Frequency masking, Cutout 등을 적용하였다.
- **비교 대상:** DS-CNN (S, M, L), TC-ResNet (8, 14 및 채널 확장 버전 1.5).

### 2. 주요 결과
- **사전 학습의 효과 (Table 1, 2):** 
  - SC 데이터셋으로 사전 학습했을 때, 레이블 데이터가 5%만 있을 때의 정확도가 $\text{TCANet}$(88.24%) $\rightarrow \text{TCANet+WVC+LGCSiam}$(92.52%)으로 크게 향상되었다.
  - AISHELL 데이터셋(다른 도메인)을 사용한 사전 학습 역시 성능 향상을 가져왔으며, 이는 제안한 표현 학습 방법이 도메인에 관계없이 범용적인 음향 특성을 추출할 수 있음을 시사한다.
- **모델 효율성 및 성능 (Table 3):** 
  - 제안된 `TCANet+WVC+LGCSiam` 모델은 **97.5%의 정확도**를 기록하며, 파라미터 수는 **65K**에 불과하다.
  - 이는 더 많은 파라미터를 가진 TC-ResNet14-1.5(305K, 96.6%)나 DS-CNN-L(20K, 95.4%)보다 높은 성능을 보이면서도 매우 효율적인 크기를 유지하고 있음을 보여준다.

## 🧠 Insights & Discussion

본 논문은 저리소스 KWS 환경에서 **"거대 모델의 지식 전이(WVC)"**와 **"자기지도 대조 학습(LGCSiam)"**의 결합이 소형 모델의 성능을 극대화할 수 있음을 입증하였다. 

특히 주목할 점은 다음과 같다.
- **데이터 효율성:** 레이블링된 데이터가 극히 적은 상황(5%)에서 사전 학습의 효과가 훨씬 두드러지게 나타났다. 이는 실제 산업 현장에서 데이터 수집 비용을 획기적으로 줄일 수 있는 가능성을 제시한다.
- **표현 학습의 보완성:** 프레임 수준의 음향 특성을 강제하는 WVC와 발화 전체의 일관성을 학습하는 LGCSiam이 서로 보완적으로 작용하여 모델의 일반화 능력을 높였다.
- **한계 및 논의:** 본 논문에서는 하드웨어 상의 실제 추론 속도(Latency)나 전력 소모에 대한 정량적 측정값은 제시되지 않았으며, 오직 파라미터 수와 정확도 위주로 분석하였다. 또한, 사용된 데이터 증강 기법들이 각 모듈의 성능에 구체적으로 어떤 영향을 미쳤는지에 대한 개별 분석(Ablation study)이 다소 부족하다.

## 📌 TL;DR

이 논문은 매우 작은 크기의 KWS 모델(TCANet, 65K params)을 위해 **Wav2Vec 2.0 기반의 지식 전이(WVC)**와 **로컬-글로벌 대조 학습(LGCSiam)**을 결합한 표현 학습 프레임워크를 제안한다. 실험 결과, 레이블링된 데이터가 부족한 상황에서도 기존의 무거운 모델들보다 더 높은 정확도(97.5%)를 달성하였으며, 이는 향후 저사양 임베디드 기기용 음성 인식 시스템 구축에 있어 효율적인 학습 전략이 될 것으로 기대된다.