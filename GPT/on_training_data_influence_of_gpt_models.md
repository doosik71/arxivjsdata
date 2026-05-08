# On Training Data Influence of GPT Models

Yekun Chai, Qingyi Liu, Shuohuan Wang, Yu Sun, Qiwei Peng, Hua Wu (2024)

## 🧩 Problem to Solve

본 논문은 생성형 언어 모델, 특히 GPT 모델의 성능에 개별 학습 데이터가 구체적으로 어떤 영향을 미치는지 분석하는 Training Data Attribution (TDA) 문제를 다룬다. 대규모 언어 모델의 성능이 급격히 향상되었음에도 불구하고, 어떤 데이터가 모델의 특정 예측이나 성능 지표에 기여했는지 정량적으로 파악하는 연구는 여전히 부족한 실정이다.

기존의 TDA 접근 방식들은 다음과 같은 한계점을 가지고 있다. 첫째, 대부분의 연구가 BERT나 T5와 같은 자연어 이해(NLU) 모델에 집중되어 있으며, 생성형 모델(Generative Models)에 대한 분석은 부족하다. 둘째, 주요 평가 지표로 Test Loss만을 사용하며, BLEU나 ROUGE와 같이 실제 생성 성능을 측정하는 지표들을 간과한다. 셋째, 기존의 시뮬레이션 기반 방식들은 학습 과정에서 본 적 없는 새로운 데이터(Unseen Data)에 대해 일반화 능력이 떨어진다는 치명적인 문제가 있다. 따라서 본 연구의 목표는 다양한 성능 지표를 예측할 수 있으며, 새로운 데이터에 대해서도 일반화가 가능한 GPT 모델 전용 데이터 영향력 분석 프레임워크인 GPTfluence를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 학습 역학(Training Dynamics)을 **특징 기반의 시뮬레이션(Featurized Simulation)**으로 모델링하는 것이다. 기존 방식들이 데이터의 인덱스나 단순한 그래디언트에 의존했던 것과 달리, GPTfluence는 사전 학습된 인코더(Pre-trained Encoder)를 사용하여 학습 및 테스트 샘플을 저차원 벡터로 임베딩한다.

이를 통해 개별 샘플 간의 상호작용을 벡터 공간에서의 연산으로 처리함으로써, 학습 시 사용되지 않은 새로운 데이터가 들어오더라도 그 특징(Feature)을 통해 영향력을 추론할 수 있는 일반화 능력을 확보하였다. 또한, 기존의 1차 마르코프 과정(1st-order Markov process)을 $n$차 마르코프 과정으로 확장하여, 현재의 성능 지표가 과거 여러 단계의 상태에 영향을 받는 복잡한 학습 역학을 더 정밀하게 포착하고자 하였다.

## 📎 Related Works

논문에서는 TDA 방법론을 크게 두 가지 갈래로 구분하여 설명한다.

1. **Gradient-Based Approximation Methods**: Influence Functions, TracIn, Grad-Dot 등이 이에 해당한다. 이들은 모델의 그래디언트 정보를 활용하여 특정 데이터의 제거 또는 추가가 테스트 결과에 미치는 영향을 근사한다. 하지만 이러한 방법들은 연산 비용이 매우 높고, 주로 Loss 값의 변화만을 추적한다는 한계가 있다.
2. **Simulation-Based Approaches**: Simfluence와 같이 학습 역학을 시뮬레이터로 학습시켜 예측하는 방식이다. Simfluence는 학습 샘플의 곱셈 및 덧셈 요소를 학습하여 테스트 손실을 예측하지만, 학습 데이터의 인덱스에 매핑된 파라미터를 학습하므로 새로운 데이터에 대한 예측이 불가능하다는 한계가 있다.

GPTfluence는 Simfluence의 시뮬레이션 방향성을 계승하면서도, 특징 기반 임베딩을 도입하여 일반화 문제를 해결하고 적용 가능한 지표의 범위를 확장했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

GPTfluence는 크게 세 단계의 파이프라인으로 구성된다.

1. **Training Dynamics Collection**: 다양한 학습 커리큘럼으로 GPT 모델을 학습시키며, 각 단계에서 테스트 샘플의 지표(Loss, BLEU, ROUGE 등) 변화를 기록하여 `GPTDynamics` 데이터셋을 구축한다.
2. **Simulator Training**: 수집된 역학 데이터를 바탕으로 특징 기반 시뮬레이터를 학습시킨다. 이때 사전 학습된 인코더를 통해 데이터를 벡터화한다.
3. **Inference**: 학습된 시뮬레이터를 사용하여 새로운 커리큘럼 하에서 특정 테스트 샘플의 성능 궤적을 자기회귀(Autoregressive) 방식으로 예측한다.

### 핵심 방정식 및 작동 원리

본 모델은 테스트 샘플 $z'$의 $t$ 시점 성능 지표 $\phi_t(z')$를 다음과 같은 $n$차 마르코프 과정으로 정의한다.

$$\phi_t(z') = \sum_{j=1}^n \alpha_j(c_t)\phi_{t-j}(z') + \beta(c_t)$$

여기서 $c_t$는 $t$ 단계에서 사용된 학습 배치이며, $\alpha_j(c_t)$는 곱셈적 영향력(Multiplicative factor), $\beta(c_t)$는 덧셈적 영향력(Additive factor)을 의미한다. 이 요소들은 배치 내 개별 샘플 $z_i$들이 가지는 영향력의 합으로 계산된다.

$$\alpha_j(c_t) = \sum_{i \in c_t} A_{i,j}, \quad \beta(c_t) = \sum_{i \in c_t} B_i$$

개별 샘플의 영향력 $A_{i,j}$와 $B_i$는 사전 학습된 인코더 $\Psi(\cdot)$를 통해 얻은 임베딩 $h_{z_i}$와 $h_{z'}$를 사용하여 다음과 같이 계산된다.

$$A_{i,j} = \langle W^{(j)\top} h_{z_i}, U^{(j)\top} h_{z'} \rangle_F$$
$$B_i = \langle W'^{\top} h_{z_i}, U'^{\top} h_{z'} \rangle_F$$

여기서 $\langle \cdot, \cdot \rangle_F$는 Frobenius 내적을 의미하며, $W$와 $U$는 학습 가능한 가중치 행렬이다. 인코더 $\Psi$는 학습 과정에서 고정(Frozen)되어 세만틱 일반화 능력을 유지한다.

### 학습 절차 및 손실 함수

시뮬레이터 $\Theta$는 실제 측정된 지표 $y_t$와 시뮬레이터가 예측한 지표 $\hat{\phi}_t$ 사이의 평균 제곱 오차(MSE)를 최소화하는 방향으로 학습되며, 과적합 방지를 위해 $L_2$ 규제항을 추가한다.

$$\Theta^\star = \arg\min_\Theta \sum_{t \in T} (y_t - \hat{\phi}_t(z'))^2 + \lambda(\|\Theta\|_2^2)$$

## 📊 Results

### 실험 설정

- **데이터셋**: FLAN 데이터셋의 일부를 사용하였으며, NLU 작업(RTE, SST-2, BoolQ)과 NLG 작업(WebNLG, WMT-16 DE/EN)을 포함한다.
- **모델**: Pythia 모델 시리즈(14M, 70M, 160M, 410M, 1B, 2.8B)를 사용하였다.
- **비교 대상**: TracIn-CP, Grad-Dot, Simfluence.
- **측정 지표**: MSE, MAE (전체 궤적), Spearman correlation coefficient $\rho$ (최종 단계).

### 주요 결과

1. **Test Loss 추정**: Instruction Tuning과 Fine-tuning 시나리오 모두에서 GPTfluence가 모든 베이스라인보다 낮은 MSE/MAE와 높은 Spearman 상관계수를 기록하였다. 특히 GPT 모델의 크기가 커져도 일관되게 우수한 성능을 유지하였다.
2. **성능 지표(BLEU, ROUGE) 추정**: 그래디언트 기반 방법론(TracIn, Grad-Dot)이 수행하지 못하는 BLEU 및 ROUGE-L 예측에서 Simfluence보다 월등한 성능을 보였다. 특히 모델 크기가 커질수록 Simfluence의 오차는 증가하는 반면, GPTfluence는 안정적인 예측 성능을 유지하였다.
3. **새로운 데이터 일반화**: 학습 시 보지 못한 새로운 학습 데이터나 테스트 데이터가 입력되었을 때도 성능 궤적을 성공적으로 시뮬레이션할 수 있음을 확인하였다.
4. **오라벨링 데이터 식별(Mislabelled Data Identification)**: SST-2 데이터셋에서 라벨이 잘못 지정된 데이터를 찾아내는 실험을 진행한 결과, 데이터 수정 후 테스트 정확도를 높이는 데 있어 Random 선택이나 TracIn-CP보다 더 효율적임을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

GPTfluence의 가장 큰 강점은 **데이터 표현의 파라미터화**를 통해 TDA의 고질적인 문제였던 일반화 능력을 확보했다는 점이다. 단순한 인덱스 매핑이 아니라 특징 공간에서의 상호작용을 모델링함으로써, 모델의 규모와 상관없이 견고한 시뮬레이션이 가능함을 보여주었다. 또한, 손실 함수뿐만 아니라 실제 서비스 지표인 BLEU, ROUGE 등을 예측할 수 있게 함으로써 실제 모델 큐레이션에 적용 가능한 실용성을 확보하였다.

### 한계 및 비판적 해석

본 연구의 가장 큰 제약 사항은 시뮬레이터를 학습시키기 위해 **방대한 양의 학습 역학 데이터(Training Dynamics)**가 필요하다는 점이다. 수백 번의 학습 런(Run)을 수행하여 데이터를 수집해야 하므로, 시뮬레이터를 구축하는 초기 비용이 매우 높다. 논문에서는 체크포인트 간격을 조정하여 이 비용을 줄이는 실험을 진행했으나, 간격이 넓어질수록 성능이 저하되는 트레이드-오프가 존재한다. 또한, 실험 대상 모델이 최대 2.8B 파라미터로 제한되어 있어, 수백억 개 이상의 파라미터를 가진 초거대 모델에서도 동일한 효율성이 유지될지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 GPT 모델의 학습 데이터 영향력을 분석하기 위해 사전 학습된 인코더와 $n$차 마르코프 과정을 결합한 특징 기반 시뮬레이터 **GPTfluence**를 제안한다. 이 방법론은 기존 TDA 방식들과 달리 **(1) Test Loss 외에 BLEU, ROUGE 등 다양한 지표를 예측할 수 있고, (2) 학습 시 보지 못한 새로운 데이터에 대해서도 일반화 능력을 가지며, (3) 모델 크기에 관계없이 일관된 성능을 보인다.** 이 연구는 향후 대규모 언어 모델의 데이터 큐레이션 최적화 및 모델 내부 동작의 투명성을 높이는 데 중요한 기여를 할 것으로 기대된다.
