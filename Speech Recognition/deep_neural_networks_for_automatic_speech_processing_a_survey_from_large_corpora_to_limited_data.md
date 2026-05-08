# Deep Neural Networks for Automatic Speech Processing: A Survey from Large Corpora to Limited Data

Vincent ROGER, Jérôme FARINAS and Julien PINQUIER (2020)

## 🧩 Problem to Solve

본 논문은 현대의 자동 음성 처리 시스템, 특히 Automatic Speech Recognition (ASR), 화자 식별(Speaker Identification), 감정 인식(Emotion Recognition) 분야에서 발생하는 **데이터 부족 문제(Under-resourced problem)**를 해결하는 것을 목표로 한다.

최신 딥러닝 기반 음성 처리 시스템(State-Of-The-Art, SOTA)은 성능 향상을 위해 방대한 양의 학습 데이터를 필요로 한다. 하지만 모든 언어나 환경에서 대규모 데이터를 확보하는 것은 불가능하며, 특히 다음과 같은 상황에서 심각한 제약이 발생한다.

- **저자원 언어(Under-resourced languages):** 디지털 리소스(음성 및 텍스트 코퍼스)가 부족하거나 언어학적 전문 지식이 부족한 경우.
- **손상된 음성(Impaired speech):** 질병(파킨슨병, 후두암 등)으로 인해 음성 신호가 변형되어 학습 데이터를 충분히 수집하기 어려운 경우.

따라서 본 논문은 대규모 데이터를 사용하는 SOTA 시스템부터 시작하여, 데이터 부족 문제를 완화하기 위한 다양한 기법들을 조사하고, 최종적으로는 소량의 데이터만으로 학습이 가능한 **Few-Shot Learning (FSL)** 기법을 음성 처리 분야에 적용할 가능성을 탐색한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 음성 처리 시스템을 데이터의 양에 따라 세 가지 단계(대규모 데이터 $\to$ 저자원 기법 $\to$ Few-Shot Learning)로 구분하여 체계적으로 분석한 서베이 보고서를 제공하는 것이다.

특히, 이미지 처리 분야에서 성공적이었던 Few-Shot Learning 프레임워크들이 음성 데이터의 특성에 어떻게 적용될 수 있는지, 그리고 화자 식별이나 손상된 음성 인식과 같은 특정 문제에 어떤 잠재적 해결책이 될 수 있는지를 이론적, 실무적 관점에서 분석하였다.

## 📎 Related Works

논문은 기존의 음성 처리 접근 방식을 다음과 같이 분류하여 설명하며 그 한계를 지적한다.

1. **SOTA ASR 시스템:** 대규모 데이터셋(예: LibriSpeech, VoxCeleb 2)을 사용하여 높은 성능을 내지만, 데이터가 적은 환경에서는 일반화 성능이 급격히 떨어진다.
2. **데이터 증강 및 도메인 전이:** noise 추가, GAN을 이용한 샘플 생성, CycleGAN을 통한 음성 개선(Enhancement) 등이 사용되나, 여전히 수동적인 설계가 필요하거나 기본 데이터셋의 의존성이 존재한다.
3. **모델 파라미터 최적화:** SincNet, LightGRU, Quaternion NN 등 파라미터 수를 줄여 Overfitting을 방지하려는 시도가 있었으나, 이는 모델 구조의 효율화일 뿐 근본적인 데이터 부족 문제를 완전히 해결하지는 못한다.
4. **전이 학습(Transfer Learning):** CPC(Contrastive Predictive Coding)와 같은 자기지도 학습(Self-supervised learning)을 통해 사전 학습된 모델을 사용한다. 그러나 소스 데이터셋과 타겟 데이터셋(예: 정상 음성 $\to$ 손상된 음성) 간의 도메인 차이가 클 경우 성능이 저하되는 한계가 있다.

## 🛠️ Methodology

본 논문은 음성 처리 시스템을 구축하는 방법론을 세 가지 관점에서 상세히 설명한다.

### 1. SOTA ASR 구조

- **Multi-models:** 음향 모델(Acoustic Model, $g$)과 언어 모델(Language Model, $f$)을 분리하여 사용한다. 최종 예측값 $\hat{y}$는 다음과 같이 정의된다.
  $$\hat{y} = f(g(x))$$
  특히 DNN-HMM 하이브리드 모델은 GMM 대신 DNN을 사용하여 음소 분류를 수행하며, Transformer 기반의 언어 모델을 결합하여 Word Error Rate (WER)를 낮춘다.
- **End-to-End (E2E) 시스템:** 입력 $x$에서 출력 $y$로 바로 매핑하는 단일 모델 $f$를 학습한다. $\hat{y} = f(x)$ 형태이며, 주로 Encoder-Decoder 구조(Stacked LSTM + Soft Attention)를 사용한다.

### 2. 저자원 대응 기법

- **SincNet:** 일반적인 1D Convolution 대신 대역폭(Bandwidth)을 나타내는 두 개의 파라미터만 사용하는 필터를 적용하여 파라미터 수를 획기적으로 줄인다.
- **LightGRU (LiGRU):** GRU에서 Reset gate를 제거하고 $\tanh$ 대신 ReLU와 Batch Normalization을 사용하여 연산 효율을 높인다.
- **Multi-task Learning:** 하나의 공유 Encoder와 여러 개의 Task-specific Decoder를 두어, 여러 작업을 동시에 학습함으로써 Encoder가 더 대표성 있는 특징을 추출하도록 유도한다.

### 3. Few-Shot Learning (FSL) 프레임워크

논문은 음성 처리에 적용 가능한 5가지 FSL 기법을 상세히 분석한다.

#### (1) Siamese Network

두 입력 $x_1, x_2$ 사이의 거리를 측정하여 유사도를 판별한다.
$$\phi(x_1, x_2) = \sigma\left(\sum \alpha |Enc(x_1) - Enc(x_2)|\right)$$
학습 시에는 다음과 같은 Contrastive Loss 형태의 손실 함수를 사용한다.
$$L = \mathbb{E}_{y(x_i)=y(\tilde{x}_j)} \log(\phi(x_i, \tilde{x}_j)) + \mathbb{E}_{y(x_i) \neq y(\tilde{x}_j)} \log(1 - \phi(x_i, \tilde{x}_j))$$

- **특징:** 화자 식별(Speaker ID) 및 검증에 유리하지만, 어휘량이 많은 ASR에는 부적합하다.

#### (2) Matching Network

Attention 메커니즘을 사용하여 Support set $S$와 Query $\hat{x}$ 사이의 관계를 학습한다.
$$\phi(\hat{x}, S) = \sum_{(x_i, y_i) \in S} a(\hat{x}, x_i)y_i$$
여기서 Attention kernel $a$는 Cosine distance와 Embedding function $f, g$를 사용하여 계산된다.

#### (3) Prototypical Networks

각 클래스 $k$에 대해 샘플들의 평균값인 프로토타입 $c_k$를 생성하고, Query와 프로토타입 간의 Euclidean distance를 측정한다.
$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f(x_i)$$
$$\phi(\hat{x}, S) = \text{softmax}_k(-d(f(\hat{x}), c_k))$$

- **특징:** 클래스당 단 한 번의 비교만 수행하므로 Siamese/Matching Network보다 연산 효율이 높다.

#### (4) Meta-Learning

학습자 모델(Trainee, $T$)과 메타 모델(Meta model, $M$)의 2단계 구조를 가진다. 메타 모델은 Trainee의 손실함수($L_T$)와 그래디언트($\nabla \theta_T$)를 입력받아 최적의 파라미터 업데이트 규칙을 학습한다.
$$\theta_{T, t}^j = M(\theta_{T, t-1}^j, L_T^j, \nabla_{\theta_{T, t-1}^j} L_T^j)$$
이는 마치 LSTM의 update gate처럼 작동하여, 매우 적은 데이터로도 빠르게 최적화된 초기 가중치를 찾게 한다.

#### (5) Graph Neural Network (GNN)

데이터 샘플들을 노드로 하는 그래프 $G=(V, E)$를 구성하고 Graph Convolution을 적용한다.
$$V^{(l+1)} = G_c(V^{(l)}, A^{(l)})$$
여기서 $A^{(l)}$는 노드 간의 거리를 기반으로 계산된 인접 행렬(Adjacency matrix)이다.

## 📊 Results

논문은 개별 실험 결과보다는 기존 문헌의 수치를 인용하여 각 방법론의 유효성을 입증한다.

- **SOTA ASR:** LibriSpeech 데이터셋에서 하이브리드 모델(DNN-HMM)은 test-clean 세트에서 **WER 2.7%**, E2E 모델은 **WER 2.44%**를 기록하며 경쟁력 있는 성능을 보였다.
- **Multi-task Learning:** 감정 인식 작업에서 성별 인식과 감정 인식을 동시에 학습했을 때, 불균형 데이터셋에서도 **전체 정확도 81.6%**라는 높은 성과를 거두었다.
- **GNN 기반 오디오 분류:** AudioSet 등을 이용한 5-way 분류 실험에서 shot 수에 따라 정확도가 **1-shot: 69.4%, 5-shot: 78.3%, 10-shot: 83.6%**로 나타나, 소량의 데이터로도 유의미한 성능 향상이 가능함을 보였다.

## 🧠 Insights & Discussion

본 논문은 다음과 같은 비판적 해석과 인사이트를 제공한다.

1. **연산 비용의 트레이드오프:** FSL 기법들은 데이터 부족 문제를 해결하지만, 클래스 수($K$)나 Shot 수($q$)가 증가할수록 연산량이 급격히 늘어난다. 따라서 수십만 개의 단어를 처리해야 하는 대규모 어휘 ASR에 그대로 적용하는 것은 현재로서는 불가능에 가깝다.
2. **현실적 적용 가능성:** 하지만 화자 식별이나 특정 질병의 음성 특징 추출과 같이 클래스 수가 제한적인 경우에는 FSL이 매우 강력한 도구가 될 수 있다. 특히, 환자에게 많은 양의 음성 데이터를 요구하는 것이 고통스러운 의료 환경에서 FSL은 필수적인 기술이 될 것이다.
3. **음성 특화 FSL의 필요성:** 이미지 분야의 FSL을 그대로 가져오는 것보다, 음성 신호의 시계열적 특성과 주파수 특성을 반영할 수 있는 Encoder 설계가 병행되어야 한다.

## 📌 TL;DR

본 논문은 방대한 데이터가 필요한 SOTA 음성 처리 시스템의 한계를 지적하고, 이를 극복하기 위한 저자원 학습 기법 및 **Few-Shot Learning (FSL)** 프레임워크(Siamese, Matching, Prototypical, Meta-Learning, GNN)를 체계적으로 분석한 서베이 논문이다. 대규모 어휘 처리에는 연산량 문제가 있으나, **손상된 음성 인식이나 화자 식별**과 같이 데이터 수집이 어려운 특수 분야에서 FSL이 혁신적인 해결책이 될 가능성이 높음을 시사한다.
