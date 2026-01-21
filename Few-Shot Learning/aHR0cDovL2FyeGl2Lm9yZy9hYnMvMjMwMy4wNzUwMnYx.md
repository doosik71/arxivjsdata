# META-LEARNING APPROACHES FOR FEW-SHOT LEARNING: A SURVEY OF RECENT ADVANCES

Hassan Gharoun, Fereshteh Momenifar, Fang Chen, Amir H. Gandomi

## 🧩 Problem to Solve

최근 심층 학습(Deep Learning, DL)은 다차원 데이터 학습에서 놀라운 성공을 거두었지만, 새로운 미지의 작업에 대해서는 성능이 저하되며, 특히 적은 수의 샘플(Few-shot datasets)에서는 일반화 능력이 크게 떨어진다는 한계가 있습니다. 이는 DL 모델이 대부분 동일한 데이터 분포 내 예측에 초점을 맞추기 때문입니다. 인간은 최소한의 관찰만으로도 새로운 개념을 학습할 수 있는 능력이 있으나, 기존 기계 학습(Machine Learning, ML) 및 DL 모델은 엄청난 양의 데이터를 필요로 합니다. 이러한 문제를 해결하기 위해, 적은 수의 데이터로 새로운 작업에 빠르게 적응하는 "학습하는 방법을 학습하는(Learning to learn)" 접근 방식인 메타 학습(Meta-learning)이 주목받고 있습니다.

## ✨ Key Contributions

이 논문은 최근 메타 학습의 발전에 대한 포괄적이고 최신 설문조사를 제공하며 다음을 기여합니다:

- 메타 학습의 기본 개념과 용어를 명확하게 소개합니다.
- 최첨단 메타 학습 방법론을 세 가지 주요 범주(거리 기반, 메모리 기반, 학습 기반)로 분류하고 각 범주 내의 최근 발전 사항을 심층적으로 분석합니다.
- 각 방법론의 세부 메커니즘과 대표적인 연구들을 상세히 설명합니다.
- 메타 학습의 현재 당면 과제와 미래 연구 방향에 대한 통찰력을 제시합니다.
- 벤치마크 데이터셋(Omniglot, MiniImageNet, CUB-200-2011)에 대한 다양한 모델들의 성능을 비교하여 제공합니다.

## 📎 Related Works

본 논문은 메타 학습의 맥락에서 다양한 선행 연구를 참조합니다. 주요 관련 연구 분야는 다음과 같습니다:

- **전통적인 기계 학습 및 심층 학습 (Traditional ML & DL):** 대규모 데이터셋에 대한 뛰어난 성능에도 불구하고, 데이터 분포 변화에 취약하고 소량의 샘플에서는 일반화가 어렵다는 한계가 있습니다.
- **전이 학습 (Transfer Learning):** 충분한 레이블 데이터가 있는 관련 소스 도메인에서 얻은 지식을 활용하여 레이블이 부족한 타겟 도메인에서 모델 성능을 향상시키는 데 사용됩니다. ImageNet 사전 학습(pre-training)된 CNN이 대표적이지만, 적은 샘플로 미세 조정할 때 성능이 저하될 수 있습니다.
- **메타 학습의 정의 (Definitions of Meta-learning):** Mitchel [13]과 Thrun and Pratt [14]에 의해 "학습하는 방법을 학습하는(Learning to learn)" 개념이 확립되었습니다.
- **기존 메타 학습 설문조사:** [16, 17]과 같은 기존 설문조사들이 있지만, 메타 학습 분야의 빠른 발전을 반영하여 더 업데이트된 리뷰를 제공합니다.

## 🛠️ Methodology

본 설문조사는 메타 학습 방법론을 세 가지 주요 범주로 분류하고 각 범주의 핵심 접근 방식을 설명합니다.

### 1. 거리 기반(Metric-based) 방법

- **핵심 아이디어:** 쿼리 세트 샘플을 서포트 세트 샘플과 비교하여 근접성을 기반으로 레이블을 예측합니다. 유사한 샘플은 임베딩 공간에서 가깝게 배치되도록 임베딩 함수를 학습하는 데 중점을 둡니다.
- **대표 모델:**
  - **Siamese Network:** 두 개의 대칭 신경망이 동일한 파라미터를 공유하며 입력 쌍의 유사성을 측정합니다. 초기 원샷(one-shot) 학습에 많이 사용되었습니다.
  - **Prototypical Network (PN):** 각 클래스의 프로토타입(평균 임베딩)을 생성하고, 쿼리 포인트와 프로토타입 간의 거리를 측정하여 분류합니다.
  - **Matching Network:** 쿼리 포인트와 서포트 세트 간의 어텐션 메커니즘을 사용하여 예측을 수행합니다.
  - **Relation Network:** 임베딩 함수와 관계 함수를 사용하여 서포트 및 쿼리 샘플 간의 관계 점수(유사도)를 학습합니다.
  - **기타:** Graph Neural Network (GNN), Global Class Representation (GCR), Multi-scale Feature Network (MSFN), Attentive Recurrent Comparators (ARCs), Region Comparison Network (RCN), Metric-agnostic Conditional Embeddings (MACO), Relative Position and Map Network (RPMN) 등이 있습니다.

### 2. 메모리 기반(Memory-based) 방법

- **핵심 아이디어:** 내부 또는 외부 메모리 컴포넌트를 사용하여 이전 입력으로부터 정보를 검색하고 동적으로 작업에 적응하는 능력을 부여합니다.
- **대표 모델:**
  - **Memory-Augmented Neural Network (MANN):** Neural Turing Machine (NTM)을 활용하여 외부 메모리에 정보를 저장하고 검색하여 작업 적응을 개선합니다.
  - **Simple Neural Attentive Meta-Learner (SNAIL):** 외부 메모리 대신 시간적 컨볼루션과 소프트 어텐션 메커니즘을 결합하여 과거 경험으로부터 정보를 활용합니다.
  - **Conditional Neural Processes (CNPs):** 메타 학습자와 작업 학습자로 구성되어 지원 세트의 간결한 표현을 사용하여 새로운 샘플의 레이블을 예측합니다.
  - **Memory Augmented Matching Network:** MANN과 Matching Network를 결합하여 왜곡된 데이터 분포에서도 편향되지 않은 클래스 프로토타입을 구축합니다.

### 3. 학습 기반(Learning-based) 방법

- **핵심 아이디어:** 메타 학습자가 기본 학습자(base learner)의 학습 프로세스 자체를 학습하거나, 초기화, 파라미터, 또는 최적화 방법을 학습합니다.
  - **초기화 학습(Learning the initialization):**
    - **Model-Agnostic Meta-Learning (MAML):** 모델 파라미터를 학습하여 새로운 미지의 작업에 대해 경사 기반 학습 규칙이 빠르게 진행되도록 합니다. 적은 수의 경사 하강 단계만으로 최적의 파라미터를 얻을 수 있는 좋은 초기화 지점을 찾습니다.
    - **기타:** PLATIPUS, BMAML, LEO, CAML, ADML, TAML, Alpha MAML, BOIL, L2F, iMAML, LLAMA 등이 있습니다.
  - **파라미터 학습(Learning the parameters):**
    - **핵심 아이디어:** 메타 학습자가 작업별 네트워크(기본 학습자)의 파라미터를 직접 생성하여 새로운 작업에 더 빠르게 적응하도록 합니다.
    - **대표 모델:** Neural Statistician, MetaNet, LGM-Net, Dynamic learning without forgetting, DAE, Weight imprinting, Incremental learning with AAN, TAFE-Net, Meta-Transfer Learner (MTL) 등이 있습니다.
  - **옵티마이저 학습(Learning the optimizer):**
    - **핵심 아이디어:** 표준 경사 하강법을 순환 신경망(RNN)과 같은 학습 가능한 옵티마이저로 대체하여 학습률이나 업데이트 규칙 자체를 학습합니다.
    - **대표 모델:** LSTM-based meta-learner, Meta-SGD, Reptile, Ridge Regression Differentiable Discriminator (R2-D2), Logistic Regression Differentiable Discriminator (LR-D2), MetaOptNet 등이 있습니다.

## 📊 Results

다양한 메타 학습 모델들의 성능은 Omniglot, MiniImageNet, CUB-200-2011 벤치마크 데이터셋에서 평가되었습니다. 주요 결과는 다음과 같습니다:

- **데이터셋 복잡도에 따른 성능 차이:**
  - **Omniglot (비교적 단순):** Meta-SGD (99.91%), Prototype-relation networks (99.90%), MAML (99.90%), Global Class Representation (99.86%) 등 많은 모델이 5-way 5-shot 문제에서 99% 이상의 높은 정확도를 달성했습니다. 이는 작업들이 밀접하게 관련되어 있을 때 대부분의 방법론이 유사하게 잘 작동함을 시사합니다.
  - **MiniImageNet (더 복잡):** 데이터셋이 복잡할수록 모델 간 성능 차이가 크게 나타났습니다.
    - **거리 기반 방법:** Multi-local feature relation network (MLFRNet)가 5-way 5-shot 문제에서 83.16%, Multi-prototype network (LMPNet)가 80.23%의 최고 정확도를 보였습니다. 이는 거리 기반 접근 방식이 국소 특징에 주목하고 클래스 내 차이를 최소화하면서 클래스 간 차이를 최대화하는 능력이 강력함을 나타냅니다.
    - **학습 기반 방법:** Model-Agnostic Meta-Learning (MAML) 기반 모델들이 전반적으로 좋은 성능을 보였으나, 계산 비용이 높은 경향이 있습니다.
- **Shot 수의 영향:** 대부분의 경우 1-shot 설정보다 5-shot 설정에서 모델 성능이 더 높게 나타나, 더 많은 샘플이 주어졌을 때 학습 효율이 증가함을 보여줍니다.

## 🧠 Insights & Discussion

메타 학습은 딥러닝의 몇 가지 주요 한계를 해결하며 유망한 발전을 이루었지만, 여전히 여러 도전 과제에 직면해 있습니다.

- **메타 학습자의 성능 불확실성:** 적은 수의 샘플로 학습하는 특성상, 관련성이 없는 작업 또는 덜 관련된 작업이 에피소드 학습(episodic training) 중에 발산하는 최적화 방향으로 이어져 예측의 불확실성을 초래할 수 있습니다.
  - **해결 방안:** 더 다양한 작업을 통해 메타 학습자를 훈련하고, 각 에피소드에서 유사한 작업을 그룹화하여 학습하는 방법을 탐색할 수 있습니다.
- **에피소드 학습의 효과적인 활용:** 에피소드 학습은 치명적인 망각(catastrophic forgetting) 문제를 야기하여 기본 클래스에 대한 모델의 과소 적합을 초래할 수 있습니다.
  - **해결 방안:** 메모리 기반 방법과 거리 기반 방법을 결합하여 기본 클래스와 새로운 클래스 모두에서 모델 성능을 향상시키는 연구가 필요합니다.
- **안정성 향상:** 거리 기반 메타 학습 알고리즘은 데이터셋에 민감하며, MAML과 같은 다른 메타 학습 알고리즘은 적대적 샘플(adversarial samples)에 대해 견고하지 못하다는 한계가 있습니다.
  - **해결 방안:** 적대적 샘플에 대한 견고성을 높이고, 다양한 데이터셋에 걸쳐 안정적인 성능을 보장하는 방법론 개발이 필요합니다.
- **다중 도메인, 다중 모드 및 교차 도메인 메타 학습:** 현재 방법론은 주로 단일 도메인에서 훈련되므로, $D_{Base}$와 $D_{Novel}$ 간의 차이가 커질수록 성능이 저하됩니다.
  - **해결 방안:** 다중 도메인 및 교차 도메인 성능을 위한 메타 학습을 개발하고, 실제 다중 모드 벤치마크 데이터셋을 구축하는 것이 중요합니다.
- **표현 학습의 효과:** 다양한 메타 학습 방법은 서로 다른 아키텍처의 임베딩 함수를 사용합니다. 임베딩 함수(예: 멀티 스케일 네트워크, ResNet)의 아키텍처 변화가 최첨단 모델의 성능 및 계산 비용에 미치는 영향을 분석하는 것이 중요합니다.
- **계산 비용 개선:** 학습 기반 방법은 적은 샘플로 학습함에도 불구하고 상당한 계산 비용이 발생할 수 있습니다. 계산 효율성을 높이는 것은 중요한 미래 연구 방향입니다.

결론적으로, 본 설문조사는 메타 학습 분야의 최신 동향과 도전 과제를 명확히 제시하며, 이 분야의 추가적인 발전을 위한 연구 방향을 제시합니다.

## 📌 TL;DR

이 논문은 딥러닝의 소량 데이터 학습 및 새로운 작업 적응 문제를 해결하는 메타 학습(Meta-learning)의 최신 동향을 포괄적으로 분석합니다. 메타 학습 방법론을 **거리 기반(Metric-based)**, **메모리 기반(Memory-based)**, 그리고 **학습 기반(Learning-based)** (초기화, 파라미터, 옵티마이저 학습) 세 가지 주요 범주로 분류하고 각 범주 내 대표 모델들을 상세히 설명합니다. 벤치마크 성능 비교를 통해 각 방법론의 강점과 약점을 제시하며, 불확실성, 치명적 망각, 안정성, 다중 도메인 학습, 표현 학습의 효과, 계산 비용 등 주요 도전 과제와 함께 미래 연구 방향을 제안하여, 인간과 유사한 "학습하는 방법을 학습하는" 지능 구현을 위한 길을 제시합니다.
