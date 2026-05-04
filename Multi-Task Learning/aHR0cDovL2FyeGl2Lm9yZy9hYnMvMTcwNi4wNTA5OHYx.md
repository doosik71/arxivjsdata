# An Overview of Multi-Task Learning in Deep Neural Networks

Sebastian Ruder (2017)

## 🧩 Problem to Solve

일반적인 머신러닝 접근 방식은 특정 벤치마크 점수나 비즈니스 KPI와 같은 단일 지표를 최적화하는 데 집중한다. 이를 위해 단일 모델이나 앙상블 모델을 훈련시키고 성능이 더 이상 오르지 않을 때까지 미세 조정(fine-tuning)을 수행한다. 그러나 이러한 단일 작업 중심의 학습은 관련 있는 다른 작업(related tasks)에서 얻을 수 있는 유용한 정보들을 무시하게 된다는 한계가 있다.

본 논문은 관련 작업들 간의 표현(representation)을 공유함으로써 모델이 원래의 주 작업(main task)에서 더 나은 일반화(generalization) 성능을 갖도록 하는 Multi-Task Learning (MTL)의 전반적인 개요를 제공하는 것을 목표로 한다. 특히 딥러닝 환경에서 MTL이 어떻게 작동하는지 설명하고, 실무자가 적절한 보조 작업(auxiliary tasks)을 선택할 수 있도록 가이드라인을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 딥러닝 기반 MTL의 이론적 배경부터 최신 방법론까지를 체계적으로 정리한 것이다. 주요 내용은 다음과 같다.

1. **MTL의 동기 부여**: 생물학적 학습, 교육적 관점, 그리고 머신러닝의 Inductive Transfer 관점에서 MTL의 필요성을 논증한다.
2. **딥러닝 MTL의 두 가지 핵심 패러다임**: 하드 파라미터 공유(Hard Parameter Sharing)와 소프트 파라미터 공유(Soft Parameter Sharing)의 구조와 차이점을 명확히 정의한다.
3. **MTL 작동 기제 분석**: 왜 MTL이 성능을 향상시키는지에 대해 Implicit Data Augmentation, Attention Focusing, Eavesdropping, Representation Bias, Regularization이라는 다섯 가지 관점에서 분석한다.
4. **보조 작업(Auxiliary Tasks) 선택 전략**: 주 작업의 성능을 높이기 위해 어떤 보조 작업을 설계해야 하는지에 대한 8가지 구체적인 전략을 제안한다.

## 📎 Related Works

논문은 신경망 이전의 선형 모델, 커널 메서드, 베이지안 알고리즘에서의 MTL 연구를 다룬다. 기존 연구들은 주로 모든 작업이 단일 출력을 갖는 Homogenous 설정에 집중했으나, 최근에는 각 작업이 고유한 출력 세트를 갖는 Heterogeneous 설정으로 확장되었다.

- **Block-sparse Regularization**: 모든 작업이 소수의 공통 특성을 공유한다는 가정하에 $\ell_1/\ell_q$ norm을 사용하여 파라미터 행렬의 희소성(sparsity)을 강제하는 방식이다.
- **Task Relationship Modeling**: 모든 작업이 밀접하게 관련되어 있지 않을 때 발생하는 Negative Transfer 문제를 해결하기 위해, 작업 간의 관계를 클러스터링하거나 트리/그래프 구조로 모델링하는 방식이다.
- **Bayesian Approaches**: 파라미터에 사전 확률 분포(prior)를 부여하거나 가우시안 프로세스(Gaussian Processes)를 확장하여 작업 간 유사성을 추론하는 방식이 제안되었다.

## 🛠️ Methodology

### 1. 딥러닝에서의 MTL 구현 방식

딥러닝에서 MTL은 크게 두 가지 파라미터 공유 방식으로 구현된다.

- **Hard Parameter Sharing**: 모든 작업이 은닉층(hidden layers)을 완전히 공유하고, 마지막 출력층(output layers)만 각 작업별로 독립적으로 가지는 구조이다. 이는 과적합(overfitting) 위험을 크게 줄이며, 공유 파라미터의 과적합 위험은 작업 수 $N$에 대해 $O(1/N)$ 수준으로 감소한다고 알려져 있다.
- **Soft Parameter Sharing**: 각 작업이 고유한 모델과 파라미터를 가지되, 서로 다른 모델의 파라미터 간 거리를 규제(regularization)하여 유사하게 유지하는 방식이다. 주로 $\ell_2$ distance나 trace norm이 사용된다.

### 2. MTL이 작동하는 내부 기제

MTL이 단일 작업 학습보다 우수한 성능을 보이는 이유는 다음과 같다.

- **Implicit Data Augmentation**: 여러 작업을 동시에 학습함으로써 데이터 샘플 크기를 실질적으로 늘리는 효과를 준다. 작업별로 서로 다른 노이즈 패턴을 가지므로, 이를 함께 학습하면 노이즈가 상쇄되어 더 일반적인 표현을 학습할 수 있다.
- **Attention Focusing**: 데이터가 고차원이고 노이즈가 많을 때, 다른 작업들이 제공하는 추가 증거를 통해 어떤 특성이 실제로 중요한지에 대한 주의(attention)를 집중할 수 있다.
- **Eavesdropping**: 어떤 특성 $G$가 작업 A에서는 학습하기 어렵지만 작업 B에서는 쉬울 수 있다. 이때 모델은 작업 B를 통해 $G$를 학습하고 이를 작업 A에 활용할 수 있다.
- **Representation Bias**: 다른 작업들이 선호하는 표현을 선택하도록 모델에 편향을 줌으로써, 향후 새로운 작업에 대해서도 더 잘 일반화될 수 있는 가설 공간을 형성한다.
- **Regularization**: Inductive Bias를 도입함으로써 모델의 Rademacher Complexity(랜덤 노이즈에 피팅하는 능력)를 낮추고 과적합을 방지한다.

### 3. 최신 딥러닝 MTL 아키텍처

- **Cross-stitch Networks**: 두 개 이상의 독립적인 네트워크를 구성하고, 각 층의 출력물에 대해 선형 결합을 학습하는 Cross-stitch unit을 배치하여 어떤 지식을 공유할지 모델이 스스로 결정하게 한다.
- **Uncertainty-based Loss Weighting**: 각 작업의 불확실성(uncertainty)을 고려하여 손실 함수의 가중치를 동적으로 조정한다. 가우시안 가능도(Gaussian likelihood)를 최대화하는 방향으로 손실 함수를 설계한다.
- **Sluice Networks**: 하드/소프트 공유, 블록 희소 규제, 작업 계층 구조를 모두 일반화한 모델이다. 어떤 층과 부분 공간(subspace)을 공유할지, 그리고 어느 층에서 최적의 표현이 학습되는지를 학습할 수 있도록 설계되었다.

## 📊 Results

본 논문은 특정 모델의 실험 결과보다는 기존 문헌들의 결과를 종합하여 분석하는 리뷰 논문이다. 다만, 다음과 같은 정성적/정량적 결론을 제시한다.

- **Hard Parameter Sharing의 한계**: 단순한 공유 방식은 작업들이 밀접하게 관련되어 있지 않거나 서로 다른 수준의 추론(reasoning level)을 요구할 때 성능이 급격히 저하된다.
- **Adaptive Sharing의 우위**: 무엇을 공유할지 모델이 직접 학습하는 방식(예: Cross-stitch, Sluice networks)이 고정된 하드 공유 방식보다 일반적으로 우수한 성능을 보인다.
- **NLP에서의 계층 구조**: NLP 작업의 경우, 품사 태깅(POS tagging)이나 개체명 인식(NER) 같은 저수준(low-level) 작업들을 네트워크의 하위 층에서 감독(supervise)하도록 설계했을 때 성능 향상이 두드러진다.
- **보조 작업의 특성**: NLP 시퀀스 태깅 문제에서 라벨 분포가 콤팩트하고 균일한 보조 작업을 사용할 때 효과적이며, 주 작업의 성능이 빠르게 정체(plateau)되는 반면 보조 작업은 정체되지 않을 때 이득이 클 가능성이 높다.

## 🧠 Insights & Discussion

### 강점 및 시사점

본 논문은 MTL을 단순히 "여러 손실 함수를 최적화하는 것"으로 정의하지 않고, 이를 통해 얻을 수 있는 Inductive Bias의 정체를 이론적으로 분석했다. 특히 보조 작업의 선택 전략을 8가지 유형으로 세분화하여 제시함으로써, 실제 구현 시 개발자가 직면하는 "어떤 작업을 추가해야 하는가"에 대한 실무적인 해답을 제공한다.

### 한계 및 비판적 해석

가장 큰 한계는 **작업 유사성(Task Similarity)**에 대한 명확한 이론적 정의가 여전히 부족하다는 점이다. 논문에서도 언급되었듯, 두 작업이 '관련 있다'는 정의가 연구자마다 다르며(특성 공유, 가설 공간 공유, 파라미터 거리 등), 이는 MTL의 성공 여부를 결정짓는 핵심 요소임에도 불구하고 여전히 경험적인 선택에 의존하고 있다.

또한, 최근의 딥러닝 모델들은 거대해지는 추세인데, 본 논문에서 다룬 파라미터 공유 방식들이 초거대 모델(LLM 등)의 효율적인 학습이나 전이 학습(Transfer Learning)과 어떻게 유기적으로 연결되는지에 대한 논의가 추가된다면 더욱 가치 있을 것이다.

## 📌 TL;DR

이 논문은 딥러닝 기반 Multi-Task Learning의 전반적인 이론과 방법론을 정리한 서베이 논문이다. MTL은 관련 작업 간의 표현을 공유하여 일반화 성능을 높이는 기법으로, 하드/소프트 파라미터 공유라는 두 가지 핵심 축을 중심으로 발전해 왔다. 저자는 MTL이 작동하는 5가지 내부 기제를 설명하고, 최신 적응형 공유 아키텍처(Sluice Networks 등)와 효과적인 보조 작업 설계 전략을 제시한다. 결과적으로, 고정된 공유 구조보다는 모델이 스스로 공유 여부를 결정하게 하는 유연한 구조가 중요하며, 향후 작업 유사성에 대한 이론적 정립이 MTL의 발전을 이끌 핵심 과제임을 시사한다.
