# MEMORY-CONSISTENT NEURAL NETWORKS FOR IMITATION LEARNING

Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, James Weimer, Insup Lee (2024)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning), 특히 전문가의 시연 데이터를 통한 지도 학습 방식인 Behavior Cloning(BC)에서 발생하는 **복합 오류(Compounding Error)** 문제를 해결하고자 한다.

Behavior Cloning은 단순한 지도 학습으로 접근할 수 있어 효율적이지만, 추론 단계에서 정책(Policy)이 훈련 데이터에 없는 생소한 상태(Unfamiliar state)에 진입할 경우 치명적인 문제가 발생한다. 정책이 아주 작은 오류를 범하면, 그 결과로 인해 전문가의 데이터 분포에서 벗어난 상태에 놓이게 되고, 이 생소한 상태에서 정책은 더 큰 오류를 범할 가능성이 높다. 이러한 오류가 시간에 따라 빠르게 누적되어 결국 전체 작업의 실패로 이어지는 것이 복합 오류 현상이다. 특히 로보틱스 분야와 같이 전문가 시연 데이터의 양이 적은 경우, Vanilla Deep Neural Networks(DNN)는 훈련 데이터 외부(Out-of-distribution)에서 매우 불안정한 출력을 생성하며, 이는 작업 수행 능력을 급격히 저하시킨다.

따라서 본 연구의 목표는 훈련 데이터 외부에서도 출력이 일정 범위 내로 제한되어 복합 오류를 억제할 수 있는 새로운 모델 클래스인 **Memory-Consistent Neural Network(MCNN)**를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델의 출력이 훈련 데이터에서 추출한 대표적인 '메모리(Memory)' 샘플들에 고정(Anchored)되어, 명시적으로 지정된 **허용 영역(Permissible regions)** 내에 머물도록 강제하는 것이다.

핵심적인 설계 직관은 다음과 같다.
1. **반-매개변수적(Semi-parametric) 구조**: 데이터셋에서 대표적인 프로토타입 샘플(Memories)을 선택하여 Scaffold로 사용하고, 그 사이를 매개변수적 DNN 함수가 보간(Interpolate)하도록 설계한다.
2. **출력의 하드 제약(Hard-constraint)**: 모델의 출력이 가장 가까운 메모리의 값에서 일정 거리($L$) 이상 벗어나지 않도록 수학적으로 제약한다.
3. **플러그인 방식의 범용성**: MCNN은 특정 아키텍처에 종속되지 않으며, MLP, Transformer, Diffusion Model 등 다양한 DNN 백본에 플러그인 형태로 적용하여 성능을 향상시킬 수 있다.

## 📎 Related Works

기존의 복합 오류 해결 방법들은 주로 학습 설정이나 데이터 수집 방식을 변경하는 방향으로 진행되었다.
- **온라인 경험 및 상호작용**: 온라인 데이터 수집(DAgger 등), 보상 라벨 활용, 전문가에게 쿼리하는 방식 등이 제안되었으나, 이는 추가적인 상호작용이나 비용이 발생한다는 한계가 있다.
- **데이터 수집 최적화**: 시연 데이터 수집 단계에서 노이즈를 주입하여 강건성을 높이는 방법 등이 연구되었다.
- **최근의 고성능 모델**: Implicit BC(IBC), Behavior Transformer(BeT), Diffusion Policy 등이 제안되었으나, 이들은 모델의 표현력을 높이는 데 집중했을 뿐, 훈련 데이터 외부에서의 출력 값을 수학적으로 제약하여 복합 오류를 방지하는 구조적 장치를 제공하지는 않는다.
- **비매개변수적 방법**: Nearest Neighbors(NN), RBF, SVM 등이 사용되었으며, 최근 VINN(Visual Imitation through Nearest Neighbors)이 제안되었다. MCNN은 이러한 NN의 안정성과 DNN의 표현력을 결합하려 한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 구조 및 구성 요소
MCNN은 **Nearest Memory Neighbor Function**과 **Constrained Neural Network Function**의 가중 합으로 구성된다.

- **Memory Code-book ($B$):** 전문가 데이터셋에서 서브샘플링된 $\{(s_i, a_i)\}_{i=1}^K$의 집합이다. 본 논문에서는 데이터의 위상을 잘 반영하기 위해 **Neural Gas** 알고리즘을 사용하여 프로토타입을 선정한다.
- **Nearest Memory Neighbor Function ($f^{NN}$):** 입력 상태 $x$에 대해 코드북 $B$에서 가장 가까운 상태 $s'$를 찾아 그에 대응하는 액션 $B(s')$를 출력하는 함수이다.
- **Constrained Neural Network ($f_\theta$):** 일반적인 DNN이며, 출력층에 압축 비선형 함수 $\sigma$ (예: $\tanh$)를 적용하여 출력을 $[-1, 1]$ 범위로 제한한다.

### 2. MCNN 방정식
입력 $x$에 대한 MCNN의 최종 출력 $f^{MC}_{\theta,B}(x)$는 다음과 같이 정의된다.

$$f^{MC}_{\theta,B}(x) = f^{NN}(x) e^{-\lambda d(x,s')} + L(1 - e^{-\lambda d(x,s')}) \sigma(f_\theta(x))$$

여기서 각 변수의 역할은 다음과 같다.
- $s' = \arg \min_{s \in B|_S} d(s, x)$: 입력 $x$와 가장 가까운 메모리 상태이다.
- $\lambda \in \mathbb{R}^+$: 메모리 근처에서 NN 함수와 DNN 함수 사이의 보간 속도를 조절하는 하이퍼파라미터이다. $\lambda=0$이면 완전한 NN 함수가 되고, $\lambda=\infty$이면 Vanilla DNN이 된다.
- $L \in \mathbb{R}$: DNN 성분이 가질 수 있는 최대 편차(Max deviation)를 결정한다.
- $\sigma$: 출력을 $[-1, 1]$로 제한하는 압축 함수이다.

결과적으로, 입력 $x$가 메모리에 매우 가까우면 $f^{NN}(x)$ 값이 지배적으로 나타나고, 메모리에서 멀어질수록 DNN의 예측값이 반영되지만, 그 값은 항상 $f^{NN}(x) \pm L$ 범위 내로 제한된다.

### 3. 학습 절차 및 손실 함수
1. **메모리 학습 (Algorithm 1)**: Neural Gas Clustering을 통해 상태 공간의 분포를 대표하는 노드들을 찾고, 각 노드에서 가장 가까운 실제 데이터 샘플을 선택하여 코드북 $B$를 생성한다.
2. **정책 학습 (Algorithm 2)**:
   - 전문가 데이터셋 $D$에 대해 표준적인 지도 학습(Supervised Learning)을 수행한다.
   - 손실 함수 $L$로는 Mean Squared Error(MSE) 또는 Negative Log-Likelihood(NLL)를 사용한다.
   - $\sigma$ 함수로는 학습 초기 단계에서 그래디언트 흐름을 돕기 위해 $\beta$ 값에 따라 변화하는 dynamic $\tanh$-like activation function을 사용한다.

### 4. 이론적 분석
논문은 MCNN의 함수 클래스 폭(Width)이 유계(Bounded)임을 증명하며, 이를 통해 sub-optimality gap에 대한 상한선(Upper bound)을 제시한다.
- **Lemma 4.6**: 함수 클래스의 폭은 $2L(1 - e^{-\lambda d^I_{B|S}})$로 상한이 정해진다. 여기서 $d^I_{B|S}$는 가장 고립된 상태(Most Isolated State)와 메모리 사이의 거리이다.
- **Theorem 4.7**: 학습된 정책 $\hat{\pi}$와 전문가 정책 $\pi^*$ 사이의 성능 차이(Sub-optimality gap)는 $\min\{H, H^2 |A| L(1 - e^{-\lambda d^I_{B|S}})\}$보다 작거나 같음을 보였다. 이는 메모리 수를 늘려 $d^I_{B|S}}$를 줄이면 이론적으로 성능 향상이 가능함을 시사한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋 및 작업**: 10가지 작업, 6가지 환경에서 평가하였다.
  - Adroit (Pen, Hammer, Relocate, Door): 정교한 로봇 조작.
  - CARLA: 자율 주행 (이미지 입력 사용).
  - Franka Kitchen: 다중 작업 수행.
- **비교 대상 (Baselines)**: Vanilla BC (MLP), 1-NN, VINN, CQL-Sparse, Implicit BC (IBC), Behavior Transformer (BeT), Diffusion BC (Diff-BC).
- **평가 지표**: 누적 보상(Cumulative Return) 및 성공 확률.

### 2. 주요 결과
- **범용적 성능 향상**: MLP, BeT, Diffusion 등 어떤 백본을 사용하더라도 MCNN 구조를 적용했을 때 성능이 일관되게 향상되었다.
- **저데이터 환경에서의 강점**: 특히 전문가 시연 데이터가 매우 적은 'Human' 데이터셋 환경에서 MCNN의 효과가 극명하게 나타났다. Figure 1에서 볼 수 있듯이, 데이터 수가 적을수록 MCNN이 Vanilla BC 대비 훨씬 높은 리턴 상승률을 보였다.
- **구체적 성과**:
  - **Adroit Human**: `MCNN+MLP`가 많은 경우 `Diffusion-BC`보다 높은 성능을 보였으며, 일부 작업(Hammer)에서는 유일하게 양수(+)의 리턴을 기록했다.
  - **CARLA**: `MCNN+MLP`가 기준선 대비 27% 성능 향상을 보였다.
  - **Franka Kitchen**: `MCNN+Diff`가 기존 `Diff-BC`보다 복잡한 작업(5개 객체 상호작용, $p_5$)에서 4배 더 높은 성공률을 보였다.

### 3. 분석 및 절제 연구 (Ablations)
- **메모리 수의 영향**: 메모리 수를 데이터셋의 10~20% 수준으로 설정했을 때 'Sweet spot'이 존재함을 확인하였다. 메모리가 너무 많아지면(100%에 가까워지면) 1-NN의 성능으로 수렴하며 성능이 오히려 저하되는 경향을 보였다.
- **메모리 선정 방식**: Neural Gas를 통한 메모리 선정이 랜덤 샘플링보다 훨씬 우수한 성능을 보였다. 이는 Neural Gas가 상태 공간의 분포를 더 고르게 캡처하여 '가장 고립된 상태'까지의 거리를 줄이기 때문이다.

## 🧠 Insights & Discussion

### 강점
MCNN은 복잡한 강화학습 알고리즘이나 추가적인 온라인 상호작용 없이, 단순한 모델 구조의 변경(Semi-parametric constraint)만으로 Behavior Cloning의 고질적인 문제인 복합 오류를 효과적으로 억제하였다. 특히, 고성능의 최신 아키텍처(Diffusion, Transformer)를 사용하더라도 MCNN의 제약 조건을 추가하는 것이 단순한 표현력 향상보다 더 중요할 수 있음을 입증하였다.

### 한계 및 논의
- **하이퍼파라미터 의존성**: $\lambda$와 $L$ 값에 따라 성능이 달라지며, 본 논문에서는 고정된 값을 사용했으나 최적의 성능을 위해서는 튜닝이 필요하다.
- **메모리 관리**: 추론 시 1-NN 검색 비용이 발생한다. 비록 본 논문에서는 그래프 기반 검색이나 GPU 병렬 처리를 통해 이를 효율적으로 해결했다고 주장하지만, 메모리 수가 극도로 많아질 경우 계산 비용이 증가할 수 있다.
- **가정**: 전문가 정책이 MCNN 함수 클래스 내에 존재한다는 'Realizability' 가정을 전제로 하며, 이는 전문가의 액션이 급격하게 변하지 않는다는 합리적인 가정에 기반한다.

## 📌 TL;DR

본 논문은 Behavior Cloning에서 발생하는 복합 오류(Compounding Error)를 해결하기 위해, 모델의 출력을 훈련 데이터의 대표 샘플(Memories) 기반 허용 영역 내로 강제하는 **Memory-Consistent Neural Networks (MCNN)**를 제안한다. MCNN은 DNN의 표현력과 Nearest Neighbor의 안정성을 결합한 형태로, MLP, Transformer, Diffusion 등 다양한 백본에 적용 가능하다. 실험 결과, 특히 데이터가 부족한 현실적인 로보틱스 환경에서 기존 BC 방식들보다 압도적인 성능 향상을 보였으며, 이는 모델의 출력 범위를 제약하는 것이 모방 학습의 강건성 확보에 핵심적임을 시사한다.