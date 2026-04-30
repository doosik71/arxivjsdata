# Multimodal Deep Learning for Low-Resource Settings: A Vector Embedding Alignment Approach for Healthcare Applications

David Restrepo et al. (2024)

## 🧩 Problem to Solve

본 논문은 저소득 및 중소득 국가(Low and Middle-Income Countries, LMICs)와 같이 컴퓨팅 자원이 제한된 환경에서 멀티모달 딥러닝 모델을 구축하고 운용하는 데 따르는 어려움을 해결하고자 한다. 일반적인 멀티모달 학습은 막대한 양의 데이터와 고성능 GPU 자원을 요구하며, 이는 자원이 부족한 지역에서 AI 기술의 격차를 심화시키는 원인이 된다.

특히 의료 분야에서는 텍스트(임상 기록)와 이미지(의료 영상) 등 서로 다른 모달리티의 데이터를 통합하는 Multimodal Data Fusion이 중요하지만, 이를 위해 raw data를 직접 처리하고 파인튜닝(fine-tuning)하는 방식은 연산 비용이 매우 높다. 따라서 본 연구의 목표는 Foundation Model에서 추출한 Vector Embedding을 활용하여 연산 효율성을 극대화하고, 의료 데이터 특유의 모달리티 간 간극을 줄이는 정렬(alignment) 방법을 제안함으로써 자원 제한 환경에서도 고성능의 멀티모달 학습을 가능하게 하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 raw data를 직접 학습시키는 대신, 이미 대규모 데이터로 사전 학습된 Foundation Model을 통해 데이터를 저차원의 벡터 임베딩으로 변환하여 처리하는 것이다. 이를 통해 모델의 파라미터 수와 연산량을 획기적으로 줄이면서도 성능을 유지할 수 있다.

또한, 본 연구는 의료 데이터가 일반 데이터에 비해 분산(variance)이 낮아 임베딩 공간에서 특정 영역에 뭉치는 'Cone Effect'가 더 심하게 나타난다는 점에 주목하였다. 이를 해결하기 위해 가우시안 노이즈 주입, 하이퍼파라미터 $\lambda$를 이용한 임베딩 시프트(shift), 그리고 정규화 손실 함수($L_{reg}$)를 통한 임베딩 정렬 방법(Embedding Alignment)을 제안하여 모달리티 간의 간극을 좁히고 성능을 향상시켰다.

## 📎 Related Works

논문에서는 BERT, GPT, LLAMA 2와 같은 NLP 모델과 ViT, DINO v2와 같은 Computer Vision 모델, 그리고 이를 통합한 CLIP, BLIP 2와 같은 Vision-Language Models(VLM)를 언급한다. 기존의 멀티모달 접근 방식은 주로 raw data를 입력으로 하여 대규모 모델을 파인튜닝하는 Transfer Learning 방식에 의존해 왔다.

그러나 이러한 방식은 저자원 환경에서 실행하기에 컴퓨팅 비용이 너무 크다는 한계가 있다. 또한 Liang et al. [44]이 제시한 'Cone Effect' 개념을 통해, 딥러닝 모델의 구조적 특성상 이미지와 텍스트 임베딩이 잠재 공간(latent space)의 매우 좁은 영역에 국한되어 서로 멀리 떨어져 배치되는 문제가 발생함을 지적한다. 본 연구는 이러한 이론적 배경을 바탕으로, 단순한 임베딩 추출을 넘어 의료 데이터에 특화된 정렬 기법을 적용함으로써 기존의 단순 퓨전 방식과 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 임베딩 추출 방식
연구팀은 세 가지 서로 다른 접근 방식을 비교 분석하였다.
- **Single-Modal Foundation Models**: 이미지 임베딩은 DINO v2를, 텍스트 임베딩은 LLAMA 2-7B(8-bit quantized)를 사용하여 개별적으로 추출한다.
- **Vision-Language Models (VLM)**: CLIP 모델을 사용하여 이미지와 텍스트의 임베딩을 추출한다. CLIP은 두 모달리티의 공동 표현(joint representation)을 학습했으므로 더 효율적일 수 있다.
- **Raw Data Approach (Baseline)**: ViT(이미지)와 BERT(텍스트) 백본을 사용하여 raw data를 직접 입력하고 파인튜닝하는 전통적인 방식을 사용한다.

### 2. Multimodal Fusion 기법
추출된 임베딩 또는 모델의 출력을 통합하기 위해 두 가지 퓨전 전략을 사용하였다.
- **Early Fusion**: 두 모달리티의 임베딩을 입력 단계에서 단순하게 연결(concatenate)한 후, Dense layer $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Batch Normalization 층으로 구성된 분류기에 통과시킨다.
- **Late-Joint Fusion**: 각 모달리티별로 독립적인 특징 추출 블록(Dense, ReLU, Dropout, BN)을 통과시킨 후, 나중에 이를 결합하여 최종 분류 헤드로 전달한다.

### 3. Embedding Alignment (모달리티 간극 감소 방법)
의료 데이터의 Cone Effect를 완화하기 위해 다음과 같은 절차를 제안한다.

**Step 1: 노이즈 주입**
임베딩을 단위 구(unit sphere) 위의 점으로 보고, 가우시안 노이즈를 추가하여 의미적 강건함(semantic robustness)을 높인다.
$$E'_{Text} = E_{Text} + \theta_t, \quad E'_{Image} = E_{Image} + \theta_i \quad (\theta \sim N(0,1))$$

**Step 2: 임베딩 시프트(Shift)**
두 모달리티 간의 평균적인 거리(Gap)를 계산하고, 하이퍼파라미터 $\lambda$를 사용하여 임베딩을 조정한다.
$$Gap = E[\|E_{Text} - E_{Image}\| | X, Y]$$
$$E'_{Text} = E_{Text} - \frac{\lambda}{2} \times Gap, \quad E'_{Image} = E_{Image} - \frac{\lambda}{2} \times Gap$$

**Step 3: 정규화 손실 함수 적용**
Late-fusion 인코더의 출력단에 다음과 같은 정규화 손실 함수를 추가하여 쌍을 이루는 샘플들을 더 가깝게 유도한다.
$$L_{reg} = \frac{1}{2N} \sum_{j=1}^{N} \|E'_{Text_j} - E'_{Image_j}\|^2_2$$

## 📊 Results

### 1. 실험 환경 및 데이터셋
- **Hardware**: GPU 없이 2 CPU cores, 64GB RAM 환경(Oracle Standard.E4.Flex)에서 수행하여 저자원 환경을 모사하였다.
- **Datasets**: BRSET(안과), HAM10000(피부과), SatelliteBench(공공보건-뎅기열 예측) 세 가지 의료 관련 데이터셋을 사용하였다.

### 2. 정량적 결과 분석
- **성능(Accuracy & F1-Score)**: BRSET 데이터셋에서 DINO v2 + LLAMA 2 조합의 Early Fusion이 정확도 0.987, F1-score 0.944로 가장 우수한 성능을 보였다. 전반적으로 임베딩 기반 방식이 Raw Data 방식보다 성능이 우수하거나 대등하였다.
- **메모리 효율성**: Raw Data 방식은 모델 크기가 약 747MB, 학습 데이터 메모리가 에포크당 최대 7.4GB에 달했으나, 임베딩 방식(CLIP 기준)은 모델 크기가 0.5MB, 학습 데이터 메모리가 50MB 수준으로 획기적으로 낮았다.
- **시간 효율성**: BRSET 기준, Raw Data 방식의 학습 시간은 에포크당 538초였으나, 임베딩 방식은 0.95~1.85초로 단축되었다. 추론 시간 또한 수백 초에서 1초 미만으로 감소하였다.

### 3. Embedding Alignment 효과
$\lambda$ 값을 조정하여 임베딩을 시프트 시킨 결과, SatelliteBench에서는 F1-score가 0.75에서 0.80으로 향상되었고, HAM10000에서는 0.715에서 0.745로 상승하는 등 유의미한 성능 향상이 관찰되었다.

## 🧠 Insights & Discussion

### 1. 강점 및 의의
본 연구는 Foundation Model의 임베딩을 활용하는 것만으로도 GPU 없이 CPU 환경에서 고성능 멀티모달 모델을 운용할 수 있음을 입증하였다. 이는 특히 의료 인프라가 부족한 LMICs 지역에서 AI 진단 도구를 보급하는 데 있어 매우 실질적인 해결책이 될 수 있다. 또한, 연산량의 극적인 감소는 탄소 배출을 줄이는 'Sustainable AI' 관점에서도 긍정적이다.

### 2. 한계 및 비판적 해석
연구에서 언급되었듯이, 일반 도메인 데이터로 학습된 Foundation Model은 의료 분야의 매우 전문적이고 기술적인 특징(domain-specific features)을 완전히 캡처하지 못할 가능성이 있다. 실제로 Raw data 파인튜닝이 이론적으로는 더 높은 잠재력을 가질 수 있으나, 본 실험에서는 데이터셋의 규모나 복잡도가 Foundation Model의 일반화 성능으로 충분히 커버 가능한 수준이었을 가능성이 있다.

또한, 제안된 $\lambda$ 시프트 방식은 휴리스틱한 접근에 가까우며, 최적의 $\lambda$ 값을 찾기 위한 추가적인 탐색 비용이 발생한다. 의료 데이터의 분산이 낮다는 점을 수학적으로 증명하고 이를 정렬 기법으로 연결한 논리는 매우 정교하지만, 다양한 의료 도메인 전체에 일반화될 수 있는지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 저자원 환경(CPU 중심)에서의 효율적인 의료 멀티모달 학습을 위해 **raw data 대신 Foundation Model의 Vector Embedding을 활용하는 방법**을 제안한다. 특히 의료 데이터에서 발생하는 **Cone Effect(모달리티 간극)**를 해결하기 위해 **노이즈 주입 및 $\lambda$-시프트 기반의 정렬 기법**을 도입하였다. 실험 결과, 메모리 사용량과 학습/추론 시간을 획기적으로 줄이면서도 성능은 오히려 향상시키거나 유지함을 확인하였으며, 이는 의료 AI의 민주화와 지속 가능한 AI 구현에 기여할 수 있는 연구이다.