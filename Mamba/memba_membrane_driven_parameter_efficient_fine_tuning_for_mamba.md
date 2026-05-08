# Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba

Donghyun Lee, Yuhangang Li, Ruokai Yin, Shiting Xiao, Priyadarshini Panda (2025)

## 🧩 Problem to Solve

본 논문은 State Space Models (SSMs), 특히 Mamba 아키텍처를 다운스트림 태스크에 적응시키기 위한 효율적인 파라미터 튜닝 방법론을 다룬다. Mamba는 Transformer 대비 선형적인 계산 복잡도를 가지며 뛰어난 확장성을 보여주지만, 모델의 크기가 커짐에 따라 모든 파라미터를 업데이트하는 Full Fine-Tuning은 막대한 계산 비용을 초래한다.

기존의 Parameter-Efficient Fine-Tuning (PEFT) 방법론들은 대부분 Transformer 아키텍처에 최적화되어 설계되었으며, 이를 Mamba에 그대로 적용할 경우 SSM 특유의 시간적 처리 역학(temporal processing dynamics)을 충분히 활용하지 못한다는 한계가 있다. 특히, Mamba의 핵심인 Selective Scan 구성 요소를 직접적으로 튜닝하는 것은 오히려 성능 저하를 일으키는 경향이 있음이 밝혀졌다. 따라서 본 연구의 목표는 pre-trained SSM의 균형 잡힌 역학을 방해하지 않으면서도, 시간적 적응 능력을 효과적으로 부여할 수 있는 Mamba 전용 PEFT 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 생물학적 뉴런의 막 전위(membrane potential) 개념에서 영감을 받은 **Leaky Integrate Membrane (LIM) 뉴런**을 Mamba의 게이팅(gating) 메커니즘에 도입하는 것이다.

주요 기여 사항은 다음과 같다:

1. **LIM 뉴런 도입**: 시간 흐름에 따라 막 전위를 축적하고 임계치에 따라 리셋하는 bio-inspired 게이팅 메커니즘을 통해 선택적 정보 유지 능력을 강화하였다.
2. **전략적 LoRA 배치 및 막 전위 전이**: Low-Rank Adaptation (LoRA)를 입력 및 출력 프로젝션 층에 최적 배치하고, 레이어 간 막 전위 상태를 전이(cross-layer membrane transfer)함으로써 네트워크 전체의 시간적 일관성을 유지하였다.
3. **효율적인 시간적 적응**: SSM의 핵심 상태 공간 구성 요소를 직접 수정하지 않고 게이팅 경로만을 강화함으로써, 적은 파라미터 업데이트만으로도 성능을 유의미하게 향상시켰다.

## 📎 Related Works

### State Space Model (SSM)

SSM은 RNN의 순환 특성과 Transformer의 병렬 처리 장점을 결합한 모델이다. S4, DSS, GSS 등을 거쳐 최근 Mamba는 데이터 의존적인 파라미터를 사용하는 Selective SSM을 도입하여 언어 모델링 및 컴퓨터 비전 분야에서 Transformer의 강력한 대안으로 자리 잡았다.

### Parameter-Efficient Fine-Tuning (PEFT)

PEFT는 전체 파라미터 중 극히 일부만 업데이트하여 효율성을 높이는 기법으로, 크게 네 가지 범주(Additive, Selective, Parameterized, Hybrid)로 나뉜다. LoRA와 같은 저차원 행렬 분해 방식이 대표적이다. 기존 Mamba 대상 PEFT 연구들은 주로 Transformer용 기법을 그대로 이식하거나 일부 프로젝션 층을 튜닝하는 수준에 그쳤으며, SSM의 구조적 특성을 반영한 시간적 제어 메커니즘의 최적화는 부족한 상태였다.

## 🛠️ Methodology

### 전체 시스템 구조

Memba는 기존 Mamba 블록의 게이팅 경로에 LIM 뉴런을 통합한 구조이다. 입력 텐서 $X_{input}$은 RMS 정규화와 입력 프로젝션($W_{in}$)을 거친 후, SSM 경로($X_{SSM}$)와 게이트 경로($X_{gate}$)로 분리된다.

- **SSM 경로**: 기존의 Selective Scan을 통해 $Y_{SSM}$을 생성한다.
- **게이트 경로**: LIM 뉴런을 통해 시간적 적응성을 부여한 후 $Y_{gate}$를 생성한다.
  $$Y_{gate} = \sigma(W_{gate\_out}(\text{LIM}(W_{gate\_in}(X_{gate})))$$
- **최종 출력**: 두 경로의 결과물을 요소별 곱(element-wise multiplication)으로 결합한 뒤 출력 프로젝션($W_{out}$)을 적용한다.
  $$Y_{out} = W_{out}(Y_{SSM} \odot Y_{gate})$$

### Leaky Integrate Membrane (LIM) 뉴런

LIM 뉴런은 입력 시퀀스를 $T$개의 청크(chunk)로 나누어 처리함으로써 계산 효율성을 확보한다. 각 청크 단계에서의 막 전위 $u$의 업데이트 식은 다음과 같다.

$$u[i+1]_l = r(\tau u[i]_l + W_l X[i])$$

여기서 $\tau \in (0, 1]$는 누설 계수(leaky factor)이며, $r(\cdot)$은 리셋 함수로 다음과 같이 정의된다.
$$r(x) = \begin{cases} 0 & \text{if } x > V_{th} \\ x & \text{otherwise} \end{cases}$$
$V_{th}$는 리셋을 결정하는 임계값이다. 이 메커니즘은 중요한 특징에 대해 높은 피크(peak)를 생성하고, 문맥이 누적됨에 따라 점진적으로 망각하는 특성을 통해 시간적 응답성을 조절한다.

### LoRA 배치 및 Cross-Layer 전이

1. **LoRA Placement**: 실험을 통해 입력 프로젝션(`in_proj`)과 출력 프로젝션(`out_proj`)에 LoRA를 적용하는 것이 가장 효과적임을 확인하였다. 이는 해당 층들이 정보의 병목 지점(bottleneck) 역할을 하기 때문이다.
2. **Cross-Layer Membrane Transfer**: 레이어 $l$에서 계산된 모든 청크의 막 전위 평균값 $\bar{u}_l$을 다음 레이어 $l+1$의 초기 상태로 전달한다.
   $$\bar{u}_l = \frac{1}{T} \sum_{i=1}^{T} u_l[i], \quad u_{l+1}[1] = \bar{u}_l$$
   이를 통해 네트워크 깊이에 관계없이 시간적 문맥이 계층적으로 흐를 수 있도록 한다.

### 이론적 배경 (Theorem 1)

본 논문은 LIM 메커니즘이 손실 함수(loss function)에 유계된 정규화 항(bounded regularization term) $R$을 추가하는 효과가 있음을 수학적으로 증명하였다.
$$E[L(\hat{y}_t)] = L(y_t \odot g(\bar{u}_t)) + R(y_t, \bar{u}_t) + O(\|\epsilon_t\|^3)$$
이 정규화 효과는 손실 평면(loss landscape)을 더 부드럽고 볼록하게(smoother and more convex) 만들어, 최적화를 용이하게 하고 일반화 성능을 높이는 결과를 가져온다.

## 📊 Results

### 실험 설정

- **언어 작업**: 8개의 상식 추론(commonsense reasoning) 벤치마크(BoolQ, PIQA, SIQA, HellaSwag 등)에서 평가하였다.
- **비전 작업**: Vim 및 VMamba 아키텍처를 사용하여 VTAB-1k 데이터셋(Natural, Specialized, Structured 도메인)에서 평가하였다.
- **비교 대상**: Full Fine-Tuning, LoRA, SLL LoRA, MambaPEFT 등.

### 정량적 결과

1. **언어 작업**: Mamba-790M 모델 기준, Memba는 기존 PEFT 방법론 대비 평균적으로 1.3%의 절대적인 성능 향상을 보였다. 특히 파라미터 효율성 측면에서 매우 적은 수의 학습 가능 파라미터만으로도 Full Fine-Tuning에 근접하거나 이를 상회하는 결과를 냈다.
2. **비전 작업**: Vim-S 아키텍처에서 `out_proj`에 LoRA를 적용한 Memba 변형이 평균 정확도 72.33%를 기록하며, 학습 파라미터 수를 28% 수준으로 줄이면서도 기존 Hybrid 방법론보다 우수한 성능을 보였다.
3. **LIM vs 기존 RNN**: LIM 뉴런을 LSTM이나 GRU로 대체했을 때보다 LIM이 더 높은 성능을 보였으며, 특히 LIM은 학습 가능한 파라미터가 전혀 없고 추론 지연 시간(latency)이 더 짧다는 강점이 있다.

### 분석 결과

- **구성 요소 영향**: LIM 뉴런, LoRA, 막 전위 전이 세 가지 요소가 모두 결합되었을 때 가장 높은 정확도를 보였다.
- **하이퍼파라미터**: 누설 계수 $\tau$가 높을수록(이전 상태를 더 많이 유지할수록) 성능이 향상되었으며, 임계값 $V_{th}=1$일 때 최적의 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 기여

Memba는 SSM의 내부 상태 공간을 건드리지 않고 게이팅 경로에 생물학적 영감을 받은 동적 메커니즘을 추가함으로써, 기존 PEFT의 한계를 극복하였다. 특히 이론적으로 증명된 손실 평면의 평활화(smoothing) 효과는 실질적인 최적화 성능 향상으로 이어졌음을 확인하였다.

### 한계 및 비판적 해석

LIM 뉴런의 본질적인 특성상 재귀적 계산(recurrent computation)이 필요하며, 이로 인해 LoRA만 적용했을 때보다 약 8.8%의 추론 시간 오버헤드가 발생한다. 비록 이 수치가 크지는 않으나, 실시간 추론이 중요한 환경에서는 부담이 될 수 있다. 하지만 저자들은 이를 Selective Scan 커널과 통합함으로써 최적화할 수 있는 가능성을 제시하였다.

또한, 막 전위의 평균값을 다음 레이어로 전달하는 방식이 정보 손실을 완전히 방지하는지에 대한 추가적인 분석이 필요해 보이나, 실험적으로는 긍정적인 결과가 도출되었다.

## 📌 TL;DR

Memba는 Mamba 모델의 시간적 적응 능력을 높이기 위해 **LIM (Leaky Integrate Membrane) 뉴런**과 전략적 **LoRA**, **레이어 간 막 전위 전이**를 결합한 PEFT 방법론이다. SSM의 핵심 구조를 수정하지 않고 게이팅 경로를 강화함으로써, 매우 적은 파라미터 업데이트만으로 언어 및 비전 작업에서 기존 PEFT 기법들을 능가하는 성능을 달성하였다. 이 연구는 향후 SSM 기반 거대 모델을 특정 태스크에 효율적으로 적응시키는 전문화된 튜닝 기법의 방향성을 제시한다.
