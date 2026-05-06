# Absolute Zero-Shot Learning

Rui Gao, Fan Wan, Daniel Organisciak, Jiyao Pu, Junyan Wang, Haoran Duan, Peng Zhang, Xingsong Hou, Yang Long (2022)

## 🧩 Problem to Solve

본 논문은 데이터 저작권 및 개인정보 보호 문제(예: 유럽의 GDPR)로 인해 실제 데이터를 공유하는 것이 점점 더 어려워지는 상황을 해결하고자 한다. 기존의 Zero-Shot Learning(ZSL) 방식은 학습 단계에서 최소한 'seen class'의 실제 이미지 데이터가 필요하며, Transductive ZSL의 경우 'unseen class'의 레이블 없는 데이터까지 요구한다. 하지만 의료나 보안 같은 전문 분야에서는 데이터 소유자(Server)와 AI 서비스 제공자(Client) 간의 데이터 공유가 불가능하거나 매우 민감한 경우가 많다.

따라서 본 연구의 목표는 학습 과정에서 **단 하나의 실제 데이터도 사용하지 않고(zero real data)** 분류기를 학습시키는 새로운 패러다임인 **Absolute Zero-Shot Learning (AZSL)**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 실제 데이터를 직접 공유하는 대신, 실제 데이터로 이미 학습된 **Teacher 모델을 데이터 보호 장치(Data Safeguard)로 활용**하여 Student 모델을 가이드하는 것이다.

1. **데이터 프리 지식 전송(Data-free Knowledge Transfer) 프레임워크**: Generator와 Student 네트워크를 통해 실제 데이터 없이도 지식을 전송하여 ZSL 및 GZSL(Generalized ZSL) 성능을 달성한다.
2. **보안 수준에 따른 시나리오 정의**: Teacher 모델의 정보 제공 수준에 따라 'White-box'와 'Black-box' 시나리오를 정의하여 보안성과 성능 간의 트레이드오프를 분석한다.
3. **Teacher 모델의 범주 확장**: Teacher 모델이 seen class만 학습했는지(Inductive), 아니면 seen과 unseen 모두 학습했는지(Transductive)에 따른 영향을 연구한다.

## 📎 Related Works

기존의 데이터 프라이버시 보호 방식인 **Federated Learning(연합 학습)**은 모델 가중치를 공유하여 데이터를 보호하지만, 이는 주로 지도 학습(Supervised Learning)에 국한되어 있으며 ZSL과 같이 학습하지 않은 클래스를 분류하는 문제로 확장하기 어렵다. 또한, 최근의 Model Inversion 기술은 공유된 모델 가중치로부터 데이터를 복원할 수 있다는 취약점이 존재한다.

**Zero-Shot Learning(ZSL)** 연구들은 시각적 특징(Visual features)과 시맨틱 임베딩(Semantic embeddings, 예: 속성이나 워드 임베딩) 간의 매핑 관계를 구축하는 데 집중해 왔다. 기존 방식들은 크게 세 가지(Visual$\rightarrow$Semantic 매핑, 데이터 생성 기반, 공통 공간 임베딩)로 나뉘지만, 공통적으로 학습 단계에서 실제 데이터(seen 또는 unseen)의 존재를 전제로 한다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조

AZSL 프레임워크는 데이터 소유자의 서버에 위치한 **Teacher 모델**과 클라이언트에 위치한 **Generator($G$)** 및 **Student 네트워크($S$)**로 구성된다. 클라이언트는 실제 데이터 대신 Generator가 생성한 가상 데이터 $\tilde{x}$를 사용하며, Teacher 모델의 가이드를 통해 이를 최적화한다.

### 학습 절차 및 주요 구성 요소

#### 1. 데이터 생성 및 검증

Generator는 노이즈 $z$와 클래스별 시맨틱 임베딩 $a$를 입력받아 가상 특징 $\tilde{x}$를 생성한다:
$$\tilde{x} = G(z|a; \theta_G)$$

생성된 데이터 중 Teacher 모델이 올바르게 분류한 고품질 샘플만 필터링하는 **데이터 검증(Data Verification)** 과정을 거친다:
$$(\tilde{x}^*, y^*) \in \{(\tilde{x}, y) | y = \text{argmax } T(\tilde{x}; \theta_T^*), \tilde{x} = G(z|a; \theta_G^*)\}$$

#### 2. 시나리오별 학습 목표

**A. White-box 시나리오 (높은 정보 공유, 낮은 보안)**
Teacher 모델이 가중치를 공유하거나 그래디언트(Gradient)를 직접 제공한다. Generator는 다음과 같은 손실 함수로 학습된다:
$$\min_{\theta_G} L(\tilde{x}, y; \theta_G) + \alpha R(\tilde{x})$$
여기서 $L$은 Teacher 모델의 Cross-entropy loss이며, $R$은 실제 데이터와 생성 데이터의 분포 거리를 최소화하는 정규화 항이다. 이후 Student 네트워크는 Teacher의 출력을 모사하도록 학습된다:
$$\min_{\theta_S} \| T^*(\tilde{x}^*; \theta_T^*) - S(\tilde{x}^*; \theta_S) \|_2^2$$

**B. Black-box 시나리오 (낮은 정보 공유, 높은 보안)**
Teacher 모델은 가중치나 그래디언트를 제공하지 않고 오직 Softmax 출력값(Pseudo labels)과 정규화 정보만 제공한다. Generator와 Student는 end-to-end로 다음과 같이 학습된다:
$$\min_{\theta_G, \theta_S} \| T^*(\tilde{x}; \theta_T^*) - S(\tilde{x}; \theta_S) \|_2^2 + \alpha R(\tilde{x})$$

### 추론 및 분류 과정

- **Transductive Teacher 사용 시**: 모든 클래스(seen, unseen)의 특징을 생성할 수 있으므로, 학습된 Student 모델을 통해 직접 클래스를 예측한다: $y^* = \text{argmax}_{y \in Y} p(y|x, \theta_S^*)$.
- **Inductive Teacher 사용 시**: Teacher가 seen class 정보만 가지고 있으므로, Generator를 이용해 unseen class의 데이터를 생성하고 이를 통해 별도의 분류기 $C$를 학습시켜 예측한다.

## 📊 Results

### 실험 설정

- **데이터셋**: AWA1, AWA2, aPY (시맨틱 정보는 BERT 임베딩 사용).
- **평가 지표**: Conventional ZSL은 unseen class에 대한 Top-1 Accuracy를 측정하며, GZSL은 seen($s$)과 unseen($u$) 정확도의 조화 평균인 Harmonic Mean $H = \frac{2 \times u \times s}{u + s}$를 사용한다.
- **구현**: ResNet101 특징(2048차원)을 입력으로 하며, 모든 네트워크는 MLP 구조를 사용한다.

### 주요 결과

- **White-box $\text{AZSL} + \text{Transductive Teacher}$**: 실제 데이터를 전혀 사용하지 않았음에도 불구하고, 기존 SOTA ZSL 모델들과 경쟁 가능한 성능을 보였다. 특히 aPY 데이터셋의 GZSL 성능(Harmonic Mean)에서 큰 폭의 상승을 보였다.
- **Black-box 시나리오**: White-box보다는 성능이 낮지만, GZSL에서 seen-unseen 클래스 간의 균형을 맞추는 능력이 뛰어남을 확인했다 (AWA2에서 unseen 성능이 seen보다 4.9% 높게 나타남).
- **Inductive Teacher의 가능성**: Teacher가 학습하지 않은 unseen class에 대해서도 Student가 새로운 지식을 생성하여 어느 정도 분류해낼 수 있음을 입증했다.
- **BERT 임베딩의 효과**: Label-conditioned나 Attribute-conditioned 방식보다 BERT 기반 시맨틱 임베딩을 사용했을 때 성능이 비약적으로 향상되었다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **데이터 부재의 극복**: 실제 데이터 없이 오직 Teacher 모델의 가이드와 시맨틱 정보만으로 고품질의 특징을 생성할 수 있음을 보였다. t-SNE 시각화 결과, 생성된 특징의 분포가 실제 데이터와 매우 유사하며 오히려 클래스별 응집도가 더 높게 나타났다.
- **편향 문제 완화**: 기존 GZSL의 고질적 문제인 'seen class로의 예측 편향'이 AZSL에서는 완화된다. 이는 seen과 unseen 클래스 모두 동일하게 '생성된 데이터'로 학습되기 때문에 일관성이 유지되기 때문이다.

### 한계 및 비판적 해석

- **Teacher 모델 의존성**: AZSL의 성능은 전적으로 Teacher 모델의 성능과 제공하는 정보(Gradient vs Output)에 의존한다.
- **보안성 논의**: White-box 시나리오에서 그래디언트 피드백이 성능을 크게 향상시키지만, 이는 동시에 Teacher 모델의 정보가 유출될 수 있는 '중위험(mid-risk)' 요소임을 명시하고 있다.
- **Inductive 설정의 성능 저하**: Inductive Teacher를 사용할 때의 성능 향상 폭이 Transductive에 비해 적으며, 이는 Teacher가 제공할 수 있는 지식의 범위가 제한적이기 때문으로 해석된다.

## 📌 TL;DR

본 논문은 실제 데이터 없이 Teacher 모델의 가이드만을 이용해 ZSL 분류기를 학습시키는 **Absolute Zero-Shot Learning (AZSL)** 패러다임을 제안한다. White-box와 Black-box라는 두 가지 보안 시나리오를 통해 성능과 보안성의 트레이드오프를 분석하였으며, 특히 White-box 설정에서는 실제 데이터를 사용한 기존 ZSL 모델에 근접하는 성능을 달성했다. 이 연구는 데이터 프라이버시가 극도로 중요한 의료, 보안 분야에서 ZSL 모델을 배포하고 학습시키는 데 중요한 기초 프레임워크를 제공한다.
