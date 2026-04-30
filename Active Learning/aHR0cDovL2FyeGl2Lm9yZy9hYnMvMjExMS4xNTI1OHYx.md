# DeepAL: Deep Active Learning in Python

Kuan-Hao Huang (2021)

## 🧩 Problem to Solve

본 논문은 기계 학습 모델 학습에 필요한 대규모 레이블링 데이터 구축 과정에서 발생하는 막대한 비용 문제를 해결하기 위한 Active Learning(능동 학습)의 구현 효율성 문제를 다룬다.

Active Learning은 모델이 학습에 가장 도움이 될 만한 데이터를 능동적으로 선택하여 레이블링 요청을 함으로써, 전체 데이터셋의 레이블을 모두 생성하지 않고도 높은 성능을 달성하는 기법이다. 기존에도 JCLAL(Java 기반)이나 libact(Python 기반)와 같은 라이브러리가 존재하였으나, 이들은 주로 Support Vector Machine(SVM)이나 Random Forest와 같은 전통적인 머신러닝 알고리즘에 최적화되어 설계되었다.

최근 딥러닝이 표준적인 학습 방법으로 자리 잡았음에도 불구하고, 딥러닝 모델의 특성을 반영하여 유연하게 사용할 수 있는 통합된 Active Learning 라이브러리가 부족하다는 점이 본 연구의 핵심 문제이다. 따라서 본 논문의 목표는 PyTorch를 기반으로 딥러닝 모델에 특화된 통합 프레임워크인 DeepAL을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 딥러닝 기반의 능동 학습(Deep Active Learning)을 위해 설계된 오픈소스 라이브러리인 DeepAL의 제안이다.

DeepAL의 중심적인 설계 아이디어는 **모듈화와 통합**이다. 사용자가 코드의 대대적인 수정 없이도 커스텀 데이터셋을 로드하고, 데이터 핸들러를 구축하며, 자신만의 Query Strategy(쿼리 전략)를 설계할 수 있도록 단순하고 통일된 프레임워크를 제공한다. 특히 PyTorch 생태계와의 호환성을 확보하여 최신 딥러닝 아키텍처를 능동 학습 파이프라인에 쉽게 접목할 수 있도록 설계되었다.

## 📎 Related Works

논문에서는 기존의 Active Learning 라이브러리로 JCLAL과 libact를 언급한다. 이러한 기존 도구들은 일반적인 능동 학습 프레임워크와 내장 쿼리 전략을 제공한다는 장점이 있다.

그러나 앞서 언급한 바와 같이, 기존 라이브러리들은 전통적인 학습 접근 방식(Traditional learning approaches)을 위해 개발되었기 때문에, 복잡한 신경망 구조를 가진 딥러닝 모델을 효율적으로 지원하지 못한다는 한계가 있다. DeepAL은 이러한 공백을 메우기 위해 PyTorch 기반의 딥러닝 전용 프레임워크를 지향하며 기존 도구들과 차별화된다.

## 🛠️ Methodology

DeepAL은 Pool-based Active Learning 설정을 기반으로 동작한다. 전체 과정은 레이블이 지정된 풀 $D_l$과 레이블이 없는 풀 $D_u$를 정의하고, 초기 분류기 $f^{(0)}$를 학습시킨 후, 매 라운드 $t$마다 쿼리 전략에 따라 $D_u$에서 $n$개의 샘플을 선택하여 $D_l$로 이동시키고 모델을 재학습시키는 반복 구조를 가진다.

시스템의 전체 아키텍처는 크게 세 가지 핵심 모듈로 구성된다.

### 1. Data 모듈
데이터셋의 관리와 전처리를 담당하는 `Data` 클래스로 구성된다.
- **데이터 관리**: 학습 풀($D_l \cup D_u$)과 테스트 세트($D_{test}$)를 유지하며, `labeledidxs`라는 이진 넘파이(numpy) 배열을 통해 각 샘플의 레이블 여부를 추적한다.
- **Data Handler**: `torch.utils.data.Dataset`을 상속받아 데이터를 텐서(Tensor) 형태로 변환하고 전처리하는 역할을 수행한다.

### 2. Net 모듈
분류기 $f$의 아키텍처와 학습 파라미터를 정의하는 `Net` 클래스로 구성된다.
- **모델 정의**: `torch.nn.Module`을 상속받은 네트워크 구조를 가지며, 은닉 표현(hidden representation)의 크기를 반환하는 `get_embeddingdim` 함수를 포함해야 한다.
- **주요 기능**: 모델 학습을 위한 `train(data)`, 예측 값을 생성하는 `predict(data)`, 확률값을 생성하는 `predict_prob(data)`, 그리고 특징 추출을 위한 `get_embeddings(data)` 함수를 제공한다.

### 3. Strategy 모듈
어떤 데이터를 레이블링할지 결정하는 쿼리 전략을 정의하는 `Strategy` 클래스로 구성된다. 
- **동작 흐름**: `query(n)` 함수를 통해 $D_u$에서 $n$개의 인덱스를 선택하고, `update(query_idxs)`를 통해 데이터 풀을 갱신하며, `train()`을 통해 모델을 업데이트한다.

#### 구현된 쿼리 전략 (Query Strategies)
DeepAL은 다음과 같은 다양한 전략들을 구현하여 제공한다.
- **Random sampling**: 무작위로 샘플을 선택한다.
- **Least confidence**: 가장 높은 확률을 가진 클래스의 확신도가 낮은 샘플을 선택한다.
  $$x^* = \arg \max_x 1 - P(\hat{y}|x)$$
- **Margin sampling**: 첫 번째와 두 번째로 확률이 높은 클래스 간의 차이가 가장 작은 샘플을 선택한다.
  $$x^* = \arg \min_x P(\hat{y}_1|x) - P(\hat{y}_2|x)$$
- **Entropy sampling**: 정보 엔트로피가 가장 높은(불확실성이 큰) 샘플을 선택한다.
  $$x^* = \arg \max_x - \sum_y P(y|x) \log P(y|x)$$
- **Uncertainty sampling with dropout estimation**: 드롭아웃(dropout)을 통해 모델의 불확실성을 추정하며, 위에서 언급한 세 가지 불확실성 지표를 모두 적용할 수 있다.
- **Bayesian active learning disagreement (BALD)**: 예측값과 모델 사후 분포(model posterior) 간의 상호 정보량(mutual information)이 최대인 샘플을 선택한다.
- **Core-set selection**: k-means 또는 k-medians 알고리즘을 기반으로 데이터 분포를 대표하는 코어셋을 선택한다.
- **Adversarial margin**: 적대적 섭동(adversarial perturbations)을 이용하여 마진을 근사하고, 마진이 가장 작은 샘플을 선택한다. (BIM 및 DeepFool 방법론 적용)

## 📊 Results

본 논문은 새로운 알고리즘을 제안하고 성능을 검증하는 연구 논문이라기보다, 소프트웨어 라이브러리를 소개하는 기술 보고서의 성격을 띠고 있다. 따라서 특정 데이터셋에 대한 정량적 벤치마크 결과나 기존 라이브러리와의 성능 비교 수치는 명시되어 있지 않다. 대신, PyTorch 기반의 통합된 인터페이스를 통해 사용자가 쉽게 AL 파이프라인을 구축할 수 있음을 의사코드(Pseudo-code)와 모듈 구조를 통해 제시하고 있다.

## 🧠 Insights & Discussion

DeepAL은 딥러닝 기반 능동 학습 연구자들에게 매우 실용적인 도구로 판단된다. 특히 다음과 같은 강점을 가진다.
첫째, PyTorch와의 긴밀한 통합을 통해 최신 신경망 아키텍처를 즉시 적용할 수 있다.
둘째, 단순한 불확실성 샘플링부터 BALD, Core-set, Adversarial 기반 전략까지 폭넓은 전략들을 하나의 인터페이스로 제공하여, 연구자가 다양한 전략을 쉽고 빠르게 비교 실험할 수 있게 한다.

다만, 라이브러리의 구현 능력에 집중되어 있어 실제 다양한 데이터셋에서 각 전략들이 어느 정도의 성능 향상을 보이는지에 대한 실험적 증명이 부족하다는 점이 한계로 지적될 수 있다. 또한, 대규모 데이터셋 환경에서 메모리 관리나 분산 학습 환경에서의 효율성에 대한 언급이 없으므로, 실제 적용 시 이 부분이 고려되어야 할 것이다.

## 📌 TL;DR

본 논문은 딥러닝 기반의 능동 학습(Deep Active Learning)을 효율적으로 구현하기 위한 PyTorch 기반의 Python 라이브러리인 **DeepAL**을 제안한다. 이 라이브러리는 데이터 관리, 모델 정의, 쿼리 전략을 모듈화하여 사용자가 커스텀 환경에서도 쉽게 능동 학습 시스템을 구축할 수 있도록 돕는다. 다양한 최신 쿼리 전략들을 내장하고 있어, 향후 딥러닝 모델의 레이블링 비용 절감을 위한 연구 및 실무 적용에 유용한 도구가 될 가능성이 높다.