# Review of Deep Learning

Zhang Rong, Li Weiping, Mo Tong (2018)

## 🧩 Problem to Solve

본 논문은 급격히 발전하고 있는 인공지능(AI)의 핵심 분야인 딥러닝(Deep Learning)의 최신 연구 동향을 체계적으로 분석하고 정리하는 것을 목표로 한다. 딥러닝은 이미 음성 처리, 컴퓨터 비전, 자연어 처리 등 다양한 분야에서 괄목할 만한 성과를 거두었으나, 여전히 학습 효율성, 모델의 신뢰성, 해석 가능성 등 해결해야 할 과제들이 많이 남아 있다. 따라서 본 논문은 딥러닝의 기본 모델부터 최신 구조적 개선 사항, 실제 적용 사례, 그리고 현재 직면한 문제점과 향후 연구 방향을 종합적으로 제시하여 해당 분야의 연구자들에게 가이드를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 주된 기여는 딥러닝의 방대한 발전 과정을 다음과 같은 논리적 흐름으로 구조화하여 분석한 점이다.

1. **기본 모델의 체계적 정리**: MLP, CNN, RNN이라는 세 가지 핵심 아키텍처의 기본 원리와 수학적 기초를 명확히 정의한다.
2. **모델 진화 과정의 추적**: AlexNet에서 ResNet에 이르는 CNN의 발전 과정과, RNN의 고질적인 문제인 Gradient Vanishing을 해결하기 위한 LSTM, GRU 등의 발전 경로를 상세히 분석한다.
3. **다학제적 응용 사례 분석**: 단순한 이미지 분류를 넘어 금융, 생물 정보학, 산업 제어 등 딥러닝이 적용된 광범위한 도메인을 분류하여 제시한다.
4. **현실적 한계와 해결책 제시**: 딥러닝의 문제를 학습(Training), 배포(Deployment), 기능(Functional), 도메인(Domain)의 네 가지 관점으로 세분화하여 분석하고, 이에 대한 구체적인 대응 방안을 제안한다.

## 📎 Related Works

논문은 딥러닝의 역사를 통해 관련 연구의 흐름을 설명한다. 1943년 McCulloch-Pitts 신경망 모델과 1958년 Rosenblatt의 Perceptron에서 시작되었으나, Minsky가 제기한 XOR 문제 해결 불가라는 한계로 인해 침체기를 겪었다. 이후 1986년 Hinton 등이 Backpropagation 알고리즘을 통해 MLP를 제안하며 비선형 분류 문제를 해결했고, 1989년 LeCun이 CNN을 통해 필기체 인식의 가능성을 보여주었다.

기존의 얕은 머신러닝(Shallow Machine Learning)과 딥러닝의 결정적인 차이점은 **Representation Learning(표현 학습)**의 여부이다. 얕은 모델은 전문가가 직접 특징을 추출하는 Feature Engineering에 의존하여 도메인 지식이 필수적이고 시간이 많이 소요되는 반면, 딥러닝은 데이터로부터 계층적인 추상 표현을 자동으로 학습하여 데이터의 고차원적 특징을 스스로 추출한다는 점에서 차별화된다.

## 🛠️ Methodology

본 논문은 딥러닝의 핵심 아키텍처 세 가지를 다음과 같이 설명한다.

### 1. Multi-Layer Perceptron (MLP)

MLP는 가장 기본적인 순전파 네트워크이다. 각 층의 뉴런은 이전 층의 출력값에 가중치를 곱하고 편향을 더한 후 비선형 활성화 함수를 통과시킨다.

- **순전파 방정식**:
  $$\text{입력 전 값: } z_{i}^{l+1} = \sum_{j} W_{ji}^{l} y_{j}^{l} + b_{i}^{l}$$
  $$\text{활성화 후 값: } y_{i}^{l+1} = f(z_{i}^{l+1})$$
- **손실 함수 (Mean Squared Error)**:
  $$J = \frac{1}{2} \sum_{i} (y_{i}^{L} - y_{i})^2$$
  여기서 $y_{i}^{L}$은 출력층의 값이고, $y_{i}$는 실제 정답 값이다. 학습의 목표는 Batch Gradient Descent 등을 통해 $J$를 최소화하는 것이다.

### 2. Convolutional Neural Networks (CNN)

공간 데이터 처리에 최적화된 구조로, Convolutional Layer와 Pooling Layer로 구성된다.

- **Convolutional Layer**: 이미지의 지역적 특징을 유지하며 특징 맵(Feature Map)을 추출한다. 가중치 공유(Weight Sharing)를 통해 파라미터 수를 줄이고 평행 이동 불변성을 제공한다.
- **Pooling Layer**: Max Pooling 또는 Average Pooling을 통해 차원을 축소하여 연산량을 줄이고 회전 불변성을 제공한다.
- **구조적 진화**: AlexNet $\rightarrow$ ZFNet $\rightarrow$ VGGNet (깊이의 중요성) $\rightarrow$ GoogLeNet (Inception 모듈을 통한 병렬 구조) $\rightarrow$ ResNet (Residual Connection을 통한 Gradient Vanishing 해결) 순으로 발전하였다.

### 3. Recurrent Neural Networks (RNN)

시계열 데이터 처리를 위해 설계되었으며, 이전 시점의 은닉 상태(Hidden State)를 현재 시점의 입력으로 다시 사용하는 기억 능력을 갖춘다.

- **순전파 방정식**:
  $$\text{은닉층 입력: } z_{h}^{t} = \sum_{i=1}^{I} w_{ih} x_{i}^{t} + \sum_{h'=1}^{H} w_{h'h} a_{h'}^{t-1}$$
  $$\text{은닉층 출력: } a_{h}^{t} = f_{h}(z_{h}^{t})$$
  $$\text{최종 출력: } y_{k}^{t} = \sum_{h=1}^{H} w_{hk} a_{h}^{t}$$
- **개선 모델**: RNN의 Gradient Vanishing 문제를 해결하기 위해 Gate 메커니즘을 도입한 LSTM(Long Short-Term Memory)과 이를 경량화한 GRU(Gated Recurrent Unit)가 제안되었다.

## 📊 Results

본 논문은 특정 실험 결과보다는 기존 연구들의 성과를 종합적으로 요약하여 제시한다.

- **컴퓨터 비전**: ImageNet 대회에서 AlexNet이 딥러닝의 가능성을 열었고, ResNet은 이미지 분류 오류율을 $3.57\%$까지 낮추어 인간의 오류율($5.1\%$)을 넘어섰다.
- **음성 처리**: Microsoft와 Google이 딥러닝을 도입하여 음성 인식 오류율을 $20\% \sim 30\%$ 수준으로 낮추는 획기적인 성과를 거두었으며, WaveNet 등을 통해 인간 수준의 음성 합성이 가능해졌다.
- **자연어 처리**: Word2Vec을 통한 단어의 벡터화(Embedding)를 시작으로 기계 번역, 질의응답(QA), 감성 분석 등에서 기존 통계 기반 모델보다 월등한 성능을 보이고 있다.
- **기타 분야**: 금융 시장 예측, 약물 분자 활성 예측, 발전소 오염 물질 배출 감소 제어 등 다양한 산업 도메인에서 딥러닝이 실질적인 성능 향상을 가져왔음을 명시한다.

## 🧠 Insights & Discussion

본 논문은 딥러닝이 거둔 성과 뒤에 숨겨진 심각한 한계점들을 날카롭게 지적하며 다음과 같은 논의를 전개한다.

**1. 학습 및 배포의 효율성 문제**
딥러닝 모델은 막대한 계산 자원(GPU/TPU)과 대규모 레이블링 데이터에 과도하게 의존한다. 특히 대규모 데이터셋 구축에 드는 비용이 너무 크기 때문에, GAN(Generative Adversarial Networks)과 같은 비지도 학습(Unsupervised Learning)으로의 전환이 필수적이다.

**2. 신뢰성과 해석 가능성 (The Black Box Problem)**
딥러닝은 높은 정확도를 보이지만, 왜 그런 결과가 나왔는지 설명할 수 없는 '블랙박스' 특성을 가진다. 또한, 입력 이미지에 미세한 노이즈를 추가하는 것만으로 오작동하게 만드는 Adversarial Attack에 취약하다는 점은 자율주행이나 의료와 같은 고신뢰 시스템 적용에 치명적인 걸림돌이 된다.

**3. 인지 능력의 부재**
현재의 딥러닝은 주로 '인지(Perception)' 단계의 작업(분류, 인식)에서 뛰어나지만, 인간과 같은 '이해(Understanding)'와 '논리적 추론(Logical Reasoning)' 능력은 부족하다. 이를 해결하기 위해 지식 그래프(Knowledge Graph)나 심볼릭 학습(Symbolic Learning)과의 결합이 필요하다고 제안한다.

**4. 데이터 효율성**
인간은 소수의 샘플만으로도 학습이 가능하지만(Few-shot Learning), 딥러닝은 수만 장의 데이터가 필요하다. 이는 딥러닝이 데이터의 통계적 패턴에 의존할 뿐, 진정한 의미의 '개념'을 학습하는 것이 아니라는 점을 시사한다.

## 📌 TL;DR

본 논문은 딥러닝의 기본 아키텍처(MLP, CNN, RNN)부터 최신 개선 모델과 다양한 응용 사례를 집대성한 종합 리뷰 보고서이다. 특히 딥러닝의 발전 과정을 체계적으로 정리함과 동시에, **학습 자원 의존성, 블랙박스 특성, 논리 추론 능력 부족**이라는 핵심 한계를 명확히 규명하였다. 향후 연구가 단순한 인식 성능 향상을 넘어, 모델 압축, 해석 가능성 확보, 그리고 심볼릭 AI와의 결합을 통한 '범용 인공지능(AGI)' 방향으로 나아가야 함을 강조하고 있다.
