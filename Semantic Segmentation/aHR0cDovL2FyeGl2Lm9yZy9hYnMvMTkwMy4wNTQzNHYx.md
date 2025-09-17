# Visual Semantic Information Pursuit: A Survey

Daqi Liu, Miroslaw Bober, Josef Kittler

## 🧩 Problem to Solve

시각적 시맨틱 정보 탐색(Visual Semantic Information Pursuit)은 컴퓨터 비전의 핵심 과제로, 각 시각적 시맨틱 단위의 의미(시각적 지각)와 이들 간의 일관된 관계(시각적 맥락 추론)를 결합하여 시각 장면을 해석하는 것을 목표로 합니다. 딥러닝의 성공으로 시각적 지각 분야는 크게 발전했지만, 시각적 지각과 시각적 맥락 추론을 통합하는 전체적인 시각 장면 시맨틱 해석 작업은 아직 초기 단계에 있습니다. 객체 탐지, 시맨틱 분할, 시각적 관계 탐지, 장면 그래프 생성 등 다양한 응용 분야의 핵심 과제임에도 불구하고, 이 흥미로운 분야에 대한 포괄적인 검토는 부족합니다.

## ✨ Key Contributions

- 모든 시각적 시맨틱 정보 탐색 방법론을 위한 통합 이론 패러다임을 제시합니다.
- 주요 발전 사항과 각 잠재적 방향에서의 미래 동향을 포괄적으로 개관합니다.
- 일반적인 벤치마크 데이터셋, 평가 지표, 그리고 해당 방법론들의 비교 결과를 소개합니다.

## 📎 Related Works

이 논문은 다음과 같은 주요 선행 연구 및 개념들을 다룹니다:

- **응용 분야**: 객체 탐지(Object Detection, OD), 시각적 시맨틱 분할(Visual Semantic Segmentation, VSS), 시각적 관계 탐지(Visual Relationship Detection, VRD), 장면 그래프 생성(Scene Graph Generation, SGG).
- **핵심 기술**:
  - **딥러닝 모델**: CNN, FCN, Faster R-CNN, RPN 등 시각적 지각 모듈에서 주로 사용됩니다.
  - **확률적 그래픽 모델**: Markov Random Fields (MRFs), Conditional Random Fields (CRFs), Undirected Cyclic Graphs (UCGs), Directed Acyclic Graphs (DAGs) 등이 시각적 맥락 추론에 활용됩니다.
  - **최적화 기법**: 변분 자유 에너지 최소화(Variational Free Energy Minimization), 메시지 패싱(Message Passing), 평균장 근사(Mean Field Approximation), 루프 신념 전파(Loopy Belief Propagation) 등.
  - **외부 지식**: Word2vec과 같은 언어학적 지식 베이스가 외부 사전 지식으로 활용됩니다.
  - **메모리 네트워크**: Neural Turing Machine (NTM), Memory-Augmented Neural Network (MANN) 등이 데이터 불균형 문제 해결을 위한 외부 메모리로 사용됩니다.
- **주요 방법론 예시**: CRF-as-RNN, Deep Parsing Network (DPN), Spatial Memory Network (SMN), Relationship Proposal Network (RPN), Visual Relationship Detection with Language Priors, Context-aware Visual Relationship Detection, Teacher-student Distillation Models.

## 🛠️ Methodology

이 논문에서 제시하는 시각적 시맨틱 정보 탐색의 통합 패러다임은 크게 두 가지 모듈로 구성됩니다:

1. **시각적 지각 모듈 (Visual Perception Module)**:

   - 입력 시각 자극으로부터 시각적 시맨틱 단위를 탐지하고 의미를 부여합니다.
   - CNN 아키텍처(FCN, Faster R-CNN 등)를 사용하여 초기 예측과 바운딩 박스 위치를 제공합니다.
   - 시각적 외관(visual appearance), 클래스 정보(class information), 상대적 공간 관계(relative spatial relationship)와 같은 사전 요소를 고려합니다.

2. **시각적 맥락 추론 모듈 (Visual Context Reasoning Module)**:
   - 탐지된 시각적 시맨틱 단위를 기반으로 최대 사후 확률(MAP) 추론을 통해 가장 가능성 높은 해석을 생성합니다.
   - 이 MAP 추론은 NP-hard 문제이므로, 선형 완화(linear relaxation) 또는 변분 자유 에너지 최소화(variational free energy minimization)를 통해 근사됩니다.
   - **사전 지식 활용**:
     - **내부 사전 지식**: 시각 자극 자체에서 얻어지는 정보 (예: 인접한 객체는 관계를 가질 가능성이 높음, 인접한 픽셀은 동일한 레이블을 가질 가능성이 높음).
     - **외부 사전 지식**: Word2vec과 같은 외부 소스(언어적 지식 베이스)에서 얻어지는 정보.
   - **최적화**: 딥러닝 기반 메시지 패싱(message passing) 전략이 MAP 추론 단계에, 확률적 경사 하강법(SGD)이 모델 선택 단계에 사용됩니다.

**통합 패러다임의 수학적 공식화 (예: 장면 그래프 생성)**:
주어진 입력 이미지 $I$와 초기 제안 바운딩 박스 $B_I$에 대해, 목표는 최적의 해석 $x^*$를 찾는 것입니다:
$$x^* = \underset{x \in \mathcal{X}}{\text{argmax}} P(x | I, B_I)$$
여기서 사후 분포 $P(x | I, B_I)$는 에너지 함수 $E(x, I, B_I)$를 통해 다음과 같이 표현될 수 있습니다:
$$P(x | I, B_I) = \frac{\exp(-E(x, I, B_I))}{Z(I, B_I)}$$
에너지 함수는 단항 포텐셜(unary potential) $\psi_u(x_i, I, B_I)$ (시각적 지각 관련)과 이항 포텐셜(binary potential) $\psi_b(x_i, x_j)$ (시각적 맥락 추론 관련)의 합으로 인수분해될 수 있습니다:
$$E(x, I, B_I) = \sum_i \psi_u(x_i, I, B_I) + \sum_{i \neq j} \psi_b(x_i, x_j)$$
이는 변분 자유 에너지 $F(\theta, Q) = \sum_{x \in \mathcal{X}} Q(x) E_\theta(x, I, B_I)$를 최소화하는 문제로 귀결됩니다.

**훈련 전략**:

- **모듈식 훈련 (Modular Training)**: 시각적 맥락 추론을 시각적 지각의 후처리 단계로 간주하며, 오류 차등이 추론 모듈 내에서만 역전파됩니다.
- **종단 간 훈련 (End-to-End Training)**: 현재 주류. 오류 차등이 이전 시각적 지각 모듈까지 역전파되어 전체 시스템을 최적화합니다. 성능 향상, 쉬운 텐서 조작, GPU를 활용한 빠른 추론 등의 장점이 있습니다.

**주요 방법론 분류**:

- **상향식 방법 (Bottom-up Methods)**: 내부 사전 지식만 사용합니다. 메시지 패싱을 활용하는 경향이 있으며, 삼중항 기반(Triplet-based), MRF/CRF 기반, 시각적 시맨틱 계층(Visual Semantic Hierarchy) 추론, DAG 기반, 외부 메모리(External Memory) 추론 모델 등으로 세분화됩니다.
- **하향식 방법 (Top-down Methods)**: 내부 및 외부(언어적) 사전 지식을 모두 통합합니다. 시맨틱 친화도(Semantic Affinity) 증류 모델, 교사-학생(Teacher-student) 증류 모델 등으로 나뉩니다. 외부 언어적 지식을 활용하여 긴 꼬리 분포(long-tail distribution) 문제를 해결하고 제로샷 학습(zero-shot learning)을 가능하게 합니다.

## 📊 Results

이 논문은 객체 탐지, 시각적 시맨틱 분할, 시각적 관계 탐지, 장면 그래프 생성의 네 가지 응용 분야에 대해 다양한 최신 방법론들의 성능을 비교합니다.

- **전반적인 추세**: 시각적 맥락 추론 모듈을 통합한 방법들이 시각적 지각 모듈에만 의존하는 방법들보다 전반적으로 우수한 성능을 보입니다. 이는 맥락 정보가 부분적으로 가려지거나 매우 작은 객체 탐지에 특히 중요함을 시사합니다.
- **객체 탐지**: ION 및 SIN과 같이 시각적 맥락 추론을 포함하는 방법이 Fast R-CNN, Faster R-CNN, YOLOv2, SSD500과 같은 순수 시각적 지각 모델보다 더 나은 mAP를 달성했습니다.
- **시각적 시맨틱 분할**: DAG-RNN + CRF는 대부분의 벤치마크(Pascal Context, Sift Flow, COCO Stuff)에서 최상의 성능을 달성했습니다. 이는 더 높은 차수의 잠재적 항을 포함하고 CRF 모듈이 객체 경계 지역화 능력을 향상시키기 때문입니다.
- **시각적 관계 탐지**: Visual Relationship Dataset에서는 CAI + SCA-M (하향식)이, Visual Genome에서는 Zoom-Net (상향식, 시각적 시맨틱 계층 추론)이 뛰어난 성능을 보였습니다. 하향식 방법은 외부 언어적 지식을 통해 긴 꼬리 분포 문제를 완화하는 반면, 상향식 방법 중 계층 추론 모델은 다양한 시맨틱 수준 간의 풍부한 맥락 전파를 통해 더 나은 최적값을 찾을 수 있습니다.
- **장면 그래프 생성**: 전역적인 맥락 정보를 통합하는 방법들(MotifNet, GPI, LinkNet)이 지역적인 맥락 정보만 사용하는 방법들보다 훨씬 뛰어난 성능을 보였으며, LinkNet이 거의 모든 평가 기준에서 가장 좋은 성능을 달성했습니다.

## 🧠 Insights & Discussion

- **시각적 맥락 추론의 중요성**: 시각적 맥락 추론은 결과 해석의 정확도와 일관성을 크게 향상시킵니다. 특히 이미지 내의 객체가 부분적으로 가려지거나 매우 작을 때 맥락 정보는 필수적입니다.
- **종단 간 훈련의 장점**: 시각적 맥락 추론 모듈의 오류 차등이 이전 시각적 지각 모듈까지 역전파될 수 있어 전체 시스템의 성능이 향상되고, 딥러닝 모델과 GPU의 병렬 처리 능력을 최대한 활용하여 추론 속도가 빨라집니다.
- **데이터 불균형 및 긴 꼬리 분포 문제**: 시각적 관계는 방대한 시맨틱 공간에 존재하며, 훈련 샘플이 부족한 긴 꼬리 분포를 따르는 경우가 많습니다. 하향식 방법은 외부 언어적 사전 지식을 증류하여 이 문제를 완화하는 데 도움을 줍니다.

**미래 연구 방향**:

1. **약지도(Weakly-supervised) 탐색 방법**: 주석 부담을 줄이기 위한 연구가 필요하지만, 현재까지 완전 지도(fully-supervised) 방법과 필적하는 성능을 달성하지 못했습니다.
2. **영역 기반 분해(Region-based Decomposition) 탐색 방법**: 기존 평균장 근사(mean field approximation)의 단순한 분해 한계를 극복하고, 고차 의사-마지널(higher-order pseudo-marginals)을 통합하여 추론 속도와 정확도를 높이는 연구가 필요합니다.
3. **고차 포텐셜 항(Higher-order Potential Terms)을 포함한 탐색 방법**: 시맨틱 일관성을 높이기 위해 고차 포텐셜 항을 통합하는 것이 중요하지만, 이는 목적 함수에 더 많은 비볼록성(non-convexities)을 주입하여 최적화 문제를 어렵게 만듭니다. 이에 대한 추가적인 노력이 필요합니다.
4. **고급 도메인 적응(Advanced Domain Adaptation)을 활용한 탐색 방법**: 데이터셋 불균형(few-shot/zero-shot learning) 문제를 해결하기 위해 관련 도메인에서 지식을 전이하는 것이 중요합니다. 현재는 주로 단항 특징(unary features) 전이에 초점을 맞추고 있으며, 구조화된 그래픽 표현(structured graphical representations)의 전이에 대한 연구가 더 필요합니다.
5. **메시지 패싱이 없는(Without Message Passing) 탐색 방법**: 현재 딥러닝 기반 시각적 시맨틱 정보 탐색 방법은 주로 메시지 패싱에 의존하지만, 이는 순차적 최적화 방법보다 경험적으로 성능이 떨어지고 실현 가능한 정수 해를 제공하지 않을 수 있습니다. 따라서 메시지 패싱 외의 다른 최적화 전략에 대한 연구가 필요합니다.

## 📌 TL;DR

시각적 시맨틱 정보 탐색은 시각적 지각과 맥락 추론을 결합하여 이미지를 깊이 이해하는 컴퓨터 비전의 핵심 과제입니다. 이 논문은 객체 탐지, 시맨틱 분할, 시각적 관계 탐지, 장면 그래프 생성을 포함한 이 분야의 최신 연구를 **통합된 딥러닝 패러다임**으로 정리합니다. 이 패러다임은 **시각적 지각 모듈**과 **시각적 맥락 추론 모듈**로 구성되며, 변분 자유 에너지 최소화를 위해 **메시지 패싱**과 같은 딥러닝 기반 최적화 전략을 사용합니다. **핵심 발견**은 맥락 추론 모듈을 통합하고 종단 간 훈련을 적용하는 것이 성능을 크게 향상시키며, 특히 외부 언어적 지식을 활용하는 하향식 방법이 데이터 불균형 문제를 완화하는 데 효과적이라는 것입니다. 또한, 장면 그래프 생성과 같은 복잡한 작업에서는 전역적 맥락 정보의 통합이 중요합니다. 향후 연구 방향으로는 약지도 학습, 고차 포텐셜 항의 도입, 고급 도메인 적응, 그리고 메시지 패싱을 대체할 새로운 최적화 방법론의 탐색이 제시됩니다.
