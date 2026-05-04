# A Selective Survey on Versatile Knowledge Distillation Paradigm for Neural Network Models

Jeong-Hoe Ku, JiHun Oh, YoungYoon Lee, Gaurav Pooniwala, SangJeong Lee (2020)

## 🧩 Problem to Solve

최근 딥러닝 모델들은 층의 깊이와 노드 수를 늘림으로써 성능을 비약적으로 향상시켰으나, 이는 필연적으로 막대한 계산 비용과 메모리 사용량이라는 문제를 야기했다. 특히 엄격한 지연 시간(latency) 요구 사항이 있는 실시간 애플리케이션을 저사양 시스템 리소스를 가진 엣지 디바이스에 배포하는 것은 매우 어렵다. 따라서 모델의 성능 저하를 최소화하면서 파라미터 크기와 계산 복잡도를 줄이는 모델 압축 기술이 필수적이다. 본 논문은 이러한 문제를 해결하기 위한 핵심 기술인 지식 증류(Knowledge Distillation, KD) 프레임워크를 체계적으로 분석하여, 연구자와 실무자들이 최적화된 모델을 개발하는 데 활용할 수 있도록 가이드를 제공하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 지식 증류를 단순히 모델 압축 도구로 보는 것이 아니라, 매우 다재다능한(versatile) 학습 패러다임으로 정의하고 이를 세 가지 핵심 요소인 '증류될 지식과 손실 함수(Distilled Knowledge and Loss)', '교사-학생 구조(Teacher-Student Paradigm)', '증류 과정(Distillation Process)'을 중심으로 분류 및 분석했다는 점이다. 또한, 컴퓨터 비전, 자연어 처리, 양자화(Quantization) 등 다양한 적용 분야뿐만 아니라 지도 학습, 약지도 학습, 준지도/비지도 학습과 같은 광범위한 딥러닝 패러다임과의 결합 가능성을 탐색하여 KD의 확장성을 입증하였다.

## 📎 Related Works

논문에서는 KD와 유사한 개념인 특권 정보 학습(Learning Using Privileged Information, LUPI)과 일반화된 증류(Generalized Distillation)를 소개한다. LUPI는 학습 단계에서만 사용할 수 있고 테스트 단계에서는 사용할 수 없는 추가적인 정보(Privileged Information)를 활용하여 학생 모델의 성능을 높이는 방식이다. 이는 KD가 단순히 모델 간의 지식 전이에 집중하는 것과 달리, 데이터의 추가적인 특성이나 설명을 활용한다는 점에서 차이가 있다. 한편, 일반화된 증류는 KD와 LUPI를 하나의 프레임워크로 통합하려는 시도로, 입력 데이터의 표현이 서로 다를 때 어떻게 지식을 전달할 것인가에 대한 이론적 분석을 제공한다.

## 🛠️ Methodology

본 논문은 KD의 내부 작동 원리를 세 가지 관점에서 상세히 설명한다.

### 1. 증류될 지식과 손실 함수 (Distilled Knowledge and Loss)
지식은 네트워크 내 어디에 존재하는가에 따라 두 가지 방식으로 정의된다.
- **응답 증류 (Response Distillation):** 교사 모델의 최종 출력값(logits)만을 사용한다. 특히 Hinton et al.은 온도 파라미터 $T$를 도입하여 Softmax 출력을 부드럽게 만든 'Soft Label'을 사용함으로써, 정답 외의 클래스 간 관계 정보인 'Dark Knowledge'를 전달하도록 했다. 이때 손실 함수로는 주로 KL Divergence를 사용한다.
- **표현 공간 증류 (Representation Space Distillation):** 교사 모델의 중간 층(hidden layers)에서 추출된 특징 맵(feature maps)이나 힌트(hints)를 모방하게 한다. 이는 모델의 구조적 차이로 인해 정보 손실이 발생할 수 있으므로, 이를 보완하기 위한 적응 층(adaptation layers) 설계가 중요하다.

### 2. 교사-학생 아키텍처 (Teacher-Student Architecture)
- **Single Teacher-Single Student:** 가장 기본적인 형태로, 하나의 큰 모델이 작은 모델을 지도한다. 다만, 두 모델 간의 용량 격차(capacity gap)가 너무 크면 성능이 오히려 저하되는 문제가 발생한다.
- **다단계 학습 (Multi-Step Learning):** 위 문제를 해결하기 위해 교사와 학생 사이에 중간 크기의 '교사 보조 모델(Teacher Assistant, TA)'을 두어 지식을 단계적으로 전달하는 TAKD 방식이 제안되었다.
- **다중 교사 학습 (Multiple-Teacher Learning):** 여러 교사 모델의 출력을 평균 내거나, 중간 층의 상대적 유사성을 제약 조건으로 부여하여 더 풍부한 지식을 전달한다.

### 3. 증류 과정 (Distillation Process)
- **오프라인 증류 (Off-line Distillation):** 미리 학습된 정적인 교사 모델로부터 학생 모델이 일방향으로 지식을 전달받는 방식이다.
- **온라인 증류 (On-line Distillation):** 교사와 학생 모델이 동시에 학습하며 서로가 서로를 가르치는 상호 학습(Mutual Learning) 방식이다. 대표적으로 Deep Mutual Learning(DML)은 여러 학생 모델이 협력하여 함께 정답을 찾아가도록 설계되었다.

## 📊 Results

본 논문은 다양한 도메인에서의 KD 적용 결과를 분석하였다.

- **컴퓨터 비전 (CV):** 이미지 분류뿐만 아니라 객체 탐지(Object Detection)에서도 성과를 보였다. 특히 FitNet과 같은 Global Feature 기반 방식과, RoI-aware 또는 Fine-grained feature imitation과 같은 Local Feature 기반 방식이 제안되었다. GAN-KD는 적대적 학습을 통해 학생 모델의 특징 맵을 교사 모델과 유사하게 만들어 성능을 높였다.
- **자연어 처리 (NLP):** BERT와 같은 거대 모델을 압축하기 위해 TinyBERT와 DistilBERT가 제안되었다. DistilBERT의 경우 모델 크기를 40% 줄이면서도 언어 이해 능력의 97%를 유지하고 속도를 60% 향상시켰다.
- **양자화 (Quantization):** 저정밀도(low-precision) 네트워크의 정확도를 높이기 위해 KD가 사용되었다. 특히 QKD(Quantization-aware KD)는 '자기 학습(Self-studying) $\rightarrow$ 공동 학습(Co-studying) $\rightarrow$ 튜터링(Tutoring)'의 3단계 과정을 통해 양자화된 모델에 최적화된 지식을 전달하여 성능을 극대화했다.

## 🧠 Insights & Discussion

본 논문은 KD가 경험적으로는 매우 성공적이지만, 이론적 배경은 여전히 부족하다는 점을 지적한다. 예를 들어, Teacher Assistant(TA) 모델을 도입했을 때 성능이 향상되는 현상은 관찰되었으나, 최적의 TA 크기가 무엇인지, 혹은 TA를 여러 층으로 쌓는 것이 효과적인지에 대한 수학적 근거는 명확하지 않다. 

또한, KD가 단순히 모델 압축을 넘어 약지도 학습(Weakly-supervised learning)에서 교차 작업 증류(cross-task distillation)로 활용되거나, 준지도 학습에서 비지도 사전 학습(unsupervised pre-training) 후 지식을 정제하는 용도로 확장되고 있다는 점은 매우 고무적이다. 이는 KD가 단순한 '모방'을 넘어 데이터 효율성을 극대화하는 '학습 가이드'로서의 역할을 수행하고 있음을 시사한다.

## 📌 TL;DR

본 논문은 지식 증류(KD)를 구성 요소, 아키텍처, 프로세스라는 세 가지 관점에서 체계적으로 분류하고, 이를 CV, NLP, 양자화 및 다양한 학습 패러다임에 적용한 사례를 분석한 종합 서베이 보고서이다. KD는 모델 압축이라는 초기 목적을 넘어, 거대 모델의 성능을 효율적으로 전이하고 학습 데이터의 부족 문제를 해결하는 범용적인 딥러닝 프레임워크로 진화하고 있으며, 향후에는 그 작동 원리에 대한 이론적 규명(Explainable KD)과 자기지도 학습과의 결합이 핵심 연구 방향이 될 것으로 전망된다.