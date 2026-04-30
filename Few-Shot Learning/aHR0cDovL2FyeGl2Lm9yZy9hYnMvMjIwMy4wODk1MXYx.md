# Meta-Learning of NAS for Few-shot Learning in Medical Image Applications

Viet-Khoa Vo-Ho, Kashu Yamazaki, Hieu Hoang, Minh-Triet Tran, and Ngan Le (2022)

## 🧩 Problem to Solve

딥러닝 모델의 성능은 네트워크 아키텍처, 하이퍼파라미터, 전/후처리 과정의 선택에 크게 의존한다. 전통적으로 이러한 설계는 연구자의 경험에 기반한 수동적인 시행착오(trial-and-error) 과정에 의존해 왔으며, 이는 막대한 시간과 계산 비용을 발생시킨다. 이를 해결하기 위해 Neural Architecture Search (NAS)가 제안되었으나, 일반적인 NAS는 여전히 다음과 같은 한계점을 가진다.

첫째, NAS는 최적의 아키텍처를 찾기 위해 대량의 레이블링된 데이터(annotated data)를 필요로 한다. 특히 의료 영상 분야는 데이터 획득 및 레이블링 비용이 매우 높아 대규모 데이터셋을 확보하기 어렵다. 둘째, NAS 과정에서 요구되는 계산 자원이 매우 방대하여 실무적인 적용에 제약이 있다. 셋째, 대부분의 NAS는 단일 작업(single task)을 위해 설계되어, 새로운 작업에 적응하려면 전체 검색 과정을 반복해야 한다.

본 논문의 목표는 의료 영상 응용 분야에서 이러한 한계를 극복하기 위해 NAS의 전반적인 메커니즘을 리뷰하고, 특히 적은 양의 데이터로도 새로운 작업에 빠르게 적응할 수 있는 Meta-learning과 NAS의 결합 방안을 분석하는 것이다.

## ✨ Key Contributions

본 보고서의 핵심 내용은 다음과 같은 세 가지 관점의 분석으로 요약된다.

1.  **NAS의 표준 구성 요소 정의**: NAS를 Search Space(탐색 공간), Search Strategy(탐색 전략), Evaluation Strategy(평가 전략)의 세 가지 핵심 구성 요소로 체계화하여 설명한다.
2.  **의료 영상 분야의 NAS 적용 사례 분석**: 의료 영상의 특성(3D 고해상도, 데이터 부족 등)을 고려하여 분류(Classification), 분할(Segmentation), 재구성(Reconstruction), 탐지(Detection) 작업에 적용된 NAS 기법들을 상세히 검토한다.
3.  **Meta-learning과 NAS의 통합(Meta-NAS)**: 고정된 아키텍처의 가중치만 학습하는 일반적인 Meta-learning과 달리, 네트워크 아키텍처와 가중치를 동시에 Meta-learning하는 메커니즘을 통해 Few-shot Learning 환경에서의 효율적인 적응 방법을 제시한다.

## 📎 Related Works

논문은 NAS의 발전 과정을 다음과 같은 관련 연구들을 통해 설명한다.

-   **초기 NAS 및 일반 CV 연구**: MetaQNN과 NAS-RL은 RL 기반의 초기 NAS 연구로, 인간의 설계를 넘어서는 효율적인 구조를 찾는 가능성을 보여주었다. NASNet은 Cell 기반의 탐색 공간을 도입하여 검색 복잡도를 낮추는 표준을 제시하였다.
-   **의료 영상 특화 NAS**: 3D-UNet, V-Net과 같은 의료 영상 표준 아키텍처를 기반으로, 이를 자동화하려는 NAS-Unet, V-NAS 등의 연구가 진행되었다. 특히 NAS-Unet은 DARTS의 미분 가능한 탐색 방식을 의료 영상 분할에 적용하였다.
-   **Meta-learning 연구**: MAML(Model-Agnostic Meta-Learning)과 REPTILE과 같은 알고리즘은 새로운 작업에 대해 소수의 그래디언트 업데이트만으로 빠르게 적응하는 가중치 학습 방법을 제시하였으며, 이는 이후 Meta-NAS의 기초가 되었다.

## 🛠️ Methodology

### 1. NAS의 일반적 수식화
NAS는 주어진 훈련 세트 $\mathcal{D}_{train}$와 검증 세트 $\mathcal{D}_{val}$에 대해, 성능 함수 $F$를 최대화하는 최적의 아키텍처 $\alpha$를 찾는 과정으로 정의된다.
$$\text{argmax}_{\alpha} = F(\mathcal{D}_{val}, \mathcal{D}_{train}, \alpha)$$

### 2. NAS의 3대 구성 요소
-   **Search Space ($\mathcal{A}$)**: 사용 가능한 연산(Convolution, Normalization, Activation 등)과 이들의 연결 구조를 정의한다. 최근에는 전체 구조를 설계하는 대신 작은 단위인 Cell(Normal Cell, Reduction Cell)을 정의하고 이를 반복적으로 쌓는 Cell-based search space가 주로 사용된다.
-   **Search Strategy**: 탐색 공간에서 후보 아키텍처를 선택하는 방법이다. Grid search, Random search부터 Gradient-based, Bayesian optimization, Evolutionary strategies (ES), Reinforcement learning (RL) 등이 있으며, 특히 ES는 유연성이 높아 널리 사용된다.
-   **Evaluation Strategy**: 선택된 아키텍처의 성능을 측정하는 방법이다. 모든 모델을 완전히 학습시키는 Full training은 비용이 너무 크기 때문에, Early-stopping, Weight-sharing(가중치 공유), Hypernetworks 등을 이용한 Partial training 방식이 제안되었다.

### 3. Meta-Learning 기반의 NAS (METANAS)
METANAS는 새로운 작업 $T_i$에 대해 최소한의 샘플만으로 최적의 아키텍처 $\alpha$와 가중치 $w$에 빠르게 적응하는 것을 목표로 한다.

**목표 함수**:
$$L(\alpha, w, p_{train}, \phi_k) = \sum_{T_i} L_{T_i}(\phi_k(\alpha, w, \mathcal{D}_{train}^{T_i}), \mathcal{D}_{val}^{T_i})$$
여기서 $\phi_k$는 $k$번의 업데이트 반복을 수행하는 Task-learner를 의미한다.

**학습 절차**:
1.  **Task-level Update (Inner Loop)**: DARTS와 같은 Task-learner를 사용하여 특정 작업 $T_i$에 최적화된 아키텍처 $\alpha^{T_i}$와 가중치 $w^{T_i}$를 찾는다. 이때 각각 $\eta_{task}^{\alpha}, \eta_{task}^{w}$의 학습률을 사용한다.
2.  **Meta-level Update (Outer Loop)**: REPTILE과 같은 Meta-learner를 사용하여, 개별 작업에서 학습된 최적 파라미터와 초기 Meta-파라미터 사이의 차이를 이용해 글로벌 Meta-architecture $\alpha$와 Meta-weights $w$를 업데이트한다. 이때 학습률 $\eta_{meta}^{\alpha}, \eta_{meta}^{w}$를 적용한다.

## 📊 Results

본 논문은 특정 단일 알고리즘의 성능을 측정하는 실험 논문이라기보다, 기존 연구들을 종합 분석한 리뷰 성격의 보고서이다. 분석된 주요 결과는 다음과 같다.

-   **이미지 분류**: Google Brain의 NAS-RL은 450개의 GPU를 사용하여 CIFAR-10 데이터셋에서 인간의 설계를 능가하는 성능을 보였으며, 의료 분야에서는 피부 병변 분류 및 fMRI 신호 분류에 적용되어 유효성을 입증하였다.
-   **의료 영상 분할**: NAS-Unet은 미분 가능한 탐색 전략과 Proxyless-NAS의 업데이트 방식을 결합하여 메모리 효율성을 높이면서도 정교한 분할 성능을 달성하였다. V-NAS는 2D, 3D, Pseudo-3D(P3D) 연산을 탐색 공간에 포함하여 데이터 특성에 맞는 최적의 연산을 스스로 선택하게 하였다.
-   **기타 응용**: MRI 재구성 분야에서는 DARTS 기반의 셀 내부 구조 탐색이 효과적이었으며, 병변 탐지(Lesion Detection)에서는 ElixirNet이 Dilated Convolution과 Non-local 연산을 포함한 탐색 공간을 통해 미세 병변 탐지 능력을 향상시켰다.

## 🧠 Insights & Discussion

### 강점 및 가능성
NAS는 인간의 편향(human bias)을 제거하여 기존의 상식을 벗어난 효율적인 구조를 발견할 수 있게 한다. 특히 Meta-learning과의 결합은 의료 영상 분야의 최대 난제인 '데이터 부족' 문제를 해결할 수 있는 강력한 도구가 된다. 소수의 데이터만으로도 빠르게 적응하는 Meta-architecture를 구축함으로써 실용적인 의료 AI 배포 가능성을 높였다.

### 한계 및 비판적 해석
1.  **여전한 인간의 개입**: Search Space 자체가 여전히 전문가가 정의한 기본 연산(primitive operations)들의 집합으로 구성되어 있어, 완전한 자동화라고 보기 어렵다.
2.  **재현성 및 비교 기준의 부재**: 각 NAS 연구마다 사용한 하이퍼파라미터, 데이터셋, 평가 프로토콜이 상이하여 서로 다른 NAS 알고리즘 간의 객관적인 성능 비교가 매우 어렵다.
3.  **자원 소모**: 효율적인 NAS가 제안되고 있음에도 불구하고, 여전히 대규모 GPU 자원이 필요하며 이는 중소 규모의 의료 기관에서 NAS를 직접 수행하는 데 큰 진입장벽이 된다.
4.  **강건성(Robustness) 문제**: 탐색된 아키텍처가 특정 데이터셋에서는 성능이 높지만, 노이즈가 섞인 데이터나 적대적 공격(adversarial attacks)에 얼마나 강건한지에 대한 분석이 부족하다.

## 📌 TL;DR

본 논문은 의료 영상 분석을 위한 NAS의 전반적인 체계와 응용 사례를 분석하고, 데이터 부족 문제를 해결하기 위해 아키텍처와 가중치를 동시에 학습하는 Meta-NAS의 메커니즘을 제시한다. 특히 METANAS와 같은 접근법은 Few-shot Learning 환경에서 의료 AI의 빠른 적응을 가능케 하며, 향후 인간의 편향이 제거된 탐색 공간 설계와 재현 가능한 평가 프로토콜 구축이 핵심 연구 방향이 될 것임을 시사한다.