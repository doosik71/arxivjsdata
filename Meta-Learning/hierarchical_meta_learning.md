# Hierarchical Meta Learning

Yingtian Zou, Jiashi Feng (2019)

## 🧩 Problem to Solve

본 논문은 기존 Meta Learning 방법론들이 학습 단계의 Task와 적용 단계의 Task가 동일한 출력 구조(Output Structure)를 공유해야 한다는 제약 조건을 해결하고자 한다. 예를 들어, $N$-개의 변수를 가진 Few-shot Regression Task로 학습된 메타 모델은 변수의 개수가 $N'$개로 변경된 새로운 Task에 직접 적용될 수 없다.

이러한 구조적 제약으로 인해, 새로운 구조를 가진 Task를 해결하기 위해서는 새로운 학습 데이터를 수집하고 시간 소모가 많은 메타 학습 과정을 처음부터 반복해야 하는 비효율성이 발생한다. 이는 범주(Category)의 수가 변하는 Few-shot Classification이나, 가능한 행동 집합이 계속해서 변하는 비정상 환경(Non-stationary environment)의 로봇 제어와 같은 실제 시나리오에서 메타 학습의 적용을 저해하는 근본적인 한계가 된다. 따라서 본 논문의 목표는 서로 다른 구조를 가진 이질적 Task(Heterogeneous Tasks)에 대해서도 효율적으로 적응하고 일반화할 수 있는 메타 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 메타 모델이 유사한 Task에 대한 적응성(Adaptability)뿐만 아니라, 서로 다른 구조를 가진 Task들 사이의 일반화 성능(Generalizability)을 명시적으로 최적화하도록 설계하는 것이다. 이를 위해 다음과 같은 기여를 제시한다.

1. **Heterogeneous Tasks를 위한 Meta Learning 문제 정의**: 기존 방법론들이 간과했던 Task 구조의 가변성 문제를 공식화하여 메타 학습의 적용 범위를 확장하였다.
2. **계층적 메타 학습(Hierarchical Meta Learning, HML) 제안**: 학습 데이터를 계층적 구조로 분해하여, 하위 수준의 적응 성능과 상위 수준의 이질적 Task 간 일반화 성능을 동시에 최적화하는 프레임워크를 제안하였다.
3. **메타 모델 변환 함수(Meta Model Transformation Function, $\omega$) 도입**: Task 구조가 변함에 따라 출력 레이어의 아키텍처가 변경되어야 하는 문제를 해결하기 위해, 내부 표현(Internal Representation)을 매끄럽게 변환해 주는 학습 가능한 변환 함수를 도입하여 적응력을 높였다.

## 📎 Related Works

기존의 Meta Learning 연구들은 주로 모델의 초기 파라미터를 최적화하여 빠른 적응을 돕는 MAML(Model-Agnostic Meta-Learning)이나, 임베딩 공간에서 쿼리 샘플과 서포트 샘플을 매칭하는 Metric Learning 기반의 Matching Networks, Prototypical Networks 등이 주를 이룬다.

그러나 이러한 접근 방식들은 모두 학습(Training)과 테스트(Testing) 단계의 Task 분포 $p(T)$가 동일하며, Task의 구조(예: 분류 클래스 개수 $N$)가 고정되어 있다는 가정을 전제로 한다. 본 논문은 이러한 가정을 깨고, 학습 시에는 $N$-way Task를 사용하더라도 테스트 시에는 $N'$-way ($N' \neq N$) Task를 해결할 수 있는 일반화 능력을 확보한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

HML은 가용 가능한 학습 Task 집합 $\mathcal{T}$를 계층적 구조 $\hat{\mathcal{T}} = \{T_1, T_2, \dots, T_H\}$로 재구성한다. 여기서 각 $T_h$는 서로 다른 구조를 가진 Task 집합이며, $T_1 \subset T_2 \subset \dots \subset T_H$의 관계를 가진다. 예를 들어, $N$-way 분류 문제에서 2-way, 3-way, ..., $N$-way 분류 Task를 순차적으로 생성하여 계층을 형성한다.

### 학습 목표 및 손실 함수

HML의 전체 목적 함수는 유사 Task에 대한 빠른 적응과 이질적 Task 간의 일반화 능력을 동시에 극대화하는 방향으로 설계되었다.

$$ \min_{\theta} \sum_{T \in \mathcal{T}} \ell(f_{\theta - \alpha \nabla_{\theta} L_T(f_{\theta})}(x_t), y_t) + \sum_{h=1}^{H-1} \sum_{T_h \in \mathcal{T}_h} L_{\theta}^{T_h \to T_{h+1}} $$

여기서 두 번째 항인 일반화 손실(Generalization Loss) $L_{\theta}^{T_h \to T_{h+1}}$는 다음과 같이 정의된다.

$$ L_{\theta}^{T_h \to T_{h+1}} = \sum_{T' \in \mathcal{T}_{h+1}} \ell(f_{\theta^{T_h} - \alpha \nabla_{\theta^{T_h}} L_{T'}(f_{\theta^{T_h}})}(x'_t), y'_t) $$

이 식은 $T_h$ 구조의 Task로 학습된 모델이 구조가 다른 $T_{h+1}$ Task에 적용되었을 때의 성능을 최적화함으로써, 모델이 특정 구조에 매몰되지 않고 광범위하게 적용 가능한 메타 지식을 학습하게 한다.

### 메타 모델 변환 함수 ($\omega$)

Task 구조가 바뀌면 출력 레이어($\phi$)의 크기가 달라져 기존 파라미터 $\theta$를 그대로 사용할 수 없다. 이를 해결하기 위해 $\theta$와 $\phi$ 사이에 변환 함수 $\omega$를 삽입한다. $\omega$는 다음과 같은 목적 함수를 통해 학습된다.

1. 메타 모델 $\theta$를 특정 Task $T_h$에 적응시켜 $\theta^h$를 얻는다: $\theta^h \leftarrow \theta - \nabla_{\theta} L_{T_h}(f_{\theta}, \phi^h)$
2. 변환 함수 $\omega$를 통해 $\theta^h$를 새로운 구조 $T_{h+1}$에 적합한 형태로 매핑한다: $\theta^{h+1} \leftarrow \theta^h - \nabla_{\theta} L_{T_{h+1}}(f(\omega(\theta^h), \phi^{h+1}))$
3. $\omega$의 손실 함수는 다음과 같다: $L_{\omega} = \sum_h \ell(f_{\omega(\theta^{h+1})}, \phi^{h+1}(x^{h+1}_t), y^{h+1}_t)$

$\omega$를 학습할 때는 다른 파라미터 $\theta$와 $\phi$를 고정하고 $\omega$만을 업데이트하여 구조 변화에 따른 최적의 매핑 규칙을 배우게 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Omniglot, miniImageNet, SUN2012 (Few-shot Classification), 그리고 Multivariate Linear Regression.
- **평가 프로토콜**: 모든 모델은 5-way Task로 학습되었으며, 테스트 시에는 $N' \ge 5$인 $N'$-way Task(Omniglot의 경우 최대 50-way)에 적용하여 일반화 성능을 측정하였다.
- **비교 대상**: 단순 Fine-tuning, MAML, Meta-SGD.

### 주요 결과

1. **Few-shot Classification**:
   - Omniglot 데이터셋에서 $N'=20, 50$인 경우, HML은 MAML 대비 각각 5.5%, 14.4% 높은 정확도를 보였다. $N'$가 커질수록 HML과 기존 방법론 간의 성능 격차가 벌어지며, 이는 HML의 강력한 일반화 능력을 입증한다.
   - miniImageNet과 SUN2012에서도 MAML보다 훨씬 완만한 성능 저하 곡선을 보이며 우수한 성능을 기록하였다.
2. **Few-shot Regression**:
   - 출력 차원 $d_y$가 5에서 20으로 증가하는 환경에서, HML은 MAML 및 Fine-tuning보다 더 빠른 적응 속도와 낮은 에러 감소율(Error Reduction Rate)을 보였다.
3. **정성적 분석**:
   - t-SNE 시각화 결과, HML로 학습된 모델의 데이터 표현(Representation)이 MAML보다 클래스 내 응집도(Intra-class compactness)가 높고 클래스 간 분리도(Inter-class separability)가 뛰어남이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 메타 학습에서 'Task 구조의 동일성'이라는 암묵적인 가정을 깨고, 이질적인 Task 간의 전이를 가능하게 했다는 점에서 학술적 가치가 크다.

특히 t-SNE 결과에서 나타나듯, HML은 단순히 파라미터를 잘 초기화하는 것을 넘어, 다양한 구조의 Task에 범용적으로 적용될 수 있는 '강건한 특징 표현(Robust Representation)'을 학습하도록 유도한다. 이는 계층적으로 구성된 Task 분포를 통해 모델이 점진적으로 복잡한 구조에 노출되면서 일반화 규칙을 습득했기 때문으로 해석된다.

다만, 변환 함수 $\omega$의 효과에 대한 분석에서 $\omega$가 없는 경우(HML w/o Trans)보다 성능이 약간 향상되기는 하지만, 그 차이가 계층적 학습 구조 자체가 주는 이득보다 작다는 점은 주목할 만하다. 이는 아키텍처의 물리적 변환보다는 학습 전략(Hierarchical Scheme)을 통한 메타 지식의 습득이 일반화 성능 향상의 주된 요인임을 시사한다.

## 📌 TL;DR

본 논문은 학습된 Task와 다른 구조(예: 클래스 개수 변경)를 가진 새로운 Task에도 빠르게 적응할 수 있는 **Hierarchical Meta Learning (HML)** 방법을 제안한다. 학습 데이터를 계층적으로 분해하여 적응성과 일반화 성능을 동시에 최적화하고, 구조 변화를 보완하는 변환 함수 $\omega$를 도입하였다. 실험을 통해 HML이 이질적인 Task 환경에서 기존 MAML 등의 방법론보다 월등한 일반화 성능을 보임을 증명하였으며, 이는 향후 가변적인 환경에서의 Few-shot Learning 및 로봇 제어 등의 분야에 중요하게 활용될 가능성이 높다.
