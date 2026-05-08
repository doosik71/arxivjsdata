# AROMA: Autonomous Rank-one Matrix Adaptation

Hao Nan Sheng, Zhi-yong Wang, Mingrui Yang, Hing Cheung So (2025)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)의 파라미터 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 과정에서 발생하는 **최적 랭크(Rank) 할당 문제**를 해결하고자 한다.

전통적인 Low-Rank Adaptation (LoRA)는 모든 레이어에 동일한 랭크를 적용하는 정적 할당 방식을 사용하는데, 이는 네트워크의 각 구성 요소가 파라미터 변화에 대해 서로 다른 민감도를 가진다는 점을 고려하지 못해 최적의 성능을 내지 못할 가능성이 크다. 이를 개선한 AdaLoRA는 SVD(특이값 분해) 기반의 동적 할당을 시도하지만, 여전히 초기 랭크와 목표 랭크 예산을 미리 설정해야 하며, Relaxed SVD 계산으로 인한 상당한 연산 오버헤드와 낮은 유효 랭크 비율(effective rank proportion)로 인한 랭크 중복 문제가 존재한다.

따라서 본 논문의 목표는 **사전 정의된 랭크 설정 없이, 각 레이어에 필요한 최적의 랭크를 자율적으로 결정하고 구축하는 효율적인 PEFT 프레임워크를 개발**하는 것이다.

## ✨ Key Contributions

AROMA의 핵심 아이디어는 기존의 '높은 랭크에서 시작해 가지치기(pruning)하는 방식'이 아니라, **'0에서 시작해 필요한 만큼 랭크를 키워나가는(rank-growing) 방식'**을 채택한 것이다.

1. **적응적 랭크 성장(Adaptive Rank Growth):** 매우 적은 수의 학습 가능 파라미터만을 사용하여 랭크-1(rank-one) 구성 요소를 점진적으로 추가함으로써 레이어별 최적 랭크를 구축한다.
2. **자율적 랭크 수렴(Automatic Rank Convergence):** 내부 루프(inner loop)는 개별 랭크-1 서브스페이스에서 정보를 추출하고, 외부 루프(outer loop)는 필요한 서브스페이스의 총 개수(즉, 최적 랭크)를 결정하는 듀얼 루프 구조를 통해 랭크를 자동 제어한다.
3. **독립적 서브스페이스 확보(Independent Subspace):** 'Check & Merge & Reinit & Reset' 전략을 통해 새로운 랭크 구성 요소를 추가할 때마다 옵티마이저 상태를 리셋함으로써, 이전 학습 경로의 간섭 없이 새로운 도메인 지식을 효율적으로 학습하도록 유도한다.

## 📎 Related Works

논문에서는 PEFT 방법론을 세 가지 패러다임으로 분류하여 설명한다.

- **Additive PEFT:** Adapter, Prefix-tuning, Prompt-tuning과 같이 보조 모듈이나 벡터를 추가하는 방식이다.
- **Selective PEFT:** BitFit, Diff pruning과 같이 모델의 특정 파라미터 부분집합만 선택적으로 업데이트하는 방식이다.
- **Reparameterized PEFT:** LoRA, AdaLoRA, DoRA 등 파라미터 공간을 변환하여 효율적인 업데이트를 수행하는 방식이다.

특히 LoRA와 AdaLoRA와의 차별점을 명확히 한다. LoRA는 정적 랭크 할당의 한계가 있고, AdaLoRA는 동적 할당을 구현했으나 초기/목표 랭크 설정에 민감하며 SVD 연산 비용이 높다는 한계가 있다. AROMA는 이러한 제약 없이 하향식(bottom-up) 성장을 통해 자율적으로 랭크를 결정함으로써 연산 효율성과 유연성을 동시에 확보한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조

AROMA는 모든 가중치 행렬 $\Delta W$를 랭크-1 행렬들의 합으로 표현한다.
$$\Delta W = \sum_{p=1}^{r} b_p a_p$$
여기서 $b_p \in \mathbb{R}^{m \times 1}$이고 $a_p \in \mathbb{R}^{1 \times n}$이다. AROMA는 한 번에 하나의 랭크-1 LoRA만 학습시키며, 학습이 완료된 구성 요소는 고정(freeze)하고 병합(merge)한 뒤 다음 랭크-1 요소를 추가한다.

### 2. 듀얼 루프 아키텍처 및 수렴 조건

- **내부 루프(Inner Loop):** 현재 단계 $p$의 랭크-1 행렬 $b_p a_p$를 학습시킨다. 다음의 수렴 조건이 충족되면 내부 루프가 종료된다.
$$\frac{\|b^{(t)}_p a^{(t)}_p\|_F - \|b^{(t-\Delta T_{in})}_p a^{(t-\Delta T_{in})}_p\|_F}{\|b^{(t-\Delta T_{in})}_p a^{(t-\Delta T_{in})}_p\|_F} < \epsilon_{in}$$
- **외부 루프(Outer Loop):** 내부 루프 종료 후, 현재 추가된 랭크-1 구성 요소가 전체 가중치 변화에 기여하는 상대적 정도를 확인하여 랭크 성장을 멈출지 결정한다.
$$\frac{\|\alpha b_p a_p\|_F}{\|W_0 + \alpha B_{p-1} A_{p-1}\|_F} < \epsilon_{out}$$
이 조건이 만족되면 해당 모듈의 랭크 학습이 완전히 종료된 것으로 간주한다.

### 3. Check & Merge & Reinit & Reset 전략

- **Check:** 위에서 언급한 내/외부 수렴 조건을 검사한다.
- **Merge & Reinit:** 수렴하지 않았다면 현재의 $b_p a_p$를 기존 행렬 $B_{p-1} A_{p-1}$에 병합하고, 새로운 $b_{p+1}, a_{p+1}$을 초기화(Kaiming initialization 등)한다.
- **Reset:** 새로운 랭크-1 구성 요소를 학습하기 전, 옵티마이저 상태(Adam의 모멘텀 등)의 99.9%를 무작위로 제거(pruning)하여 리셋한다. 이는 옵티마이저가 이전의 최적화 경로에 갇히지 않고 새로운 서브스페이스를 탐색하게 하기 위함이다.
- **Warmup:** 새로운 랭크 구성 요소가 추가될 때마다 짧은 웜업 단계를 거쳐 초기 오버피팅을 방지한다.

## 📊 Results

### 1. 실험 설정

- **모델 및 데이터셋:**
  - NLU 태스크: RoBERTa-base $\to$ GLUE 벤치마크.
  - 상식 추론(Commonsense Reasoning) 태스크: LLaMA3-8B $\to$ Commonsense170K 데이터셋.
- **비교 대상:** Full Fine-tuning, BitFit, Adapter-H, Adapter-P, LoRA, AdaLoRA, ReLoRA.
- **측정 지표:** Accuracy, Matthew's Correlation Coefficient (MC), Pearson Correlation Coefficient (PC).

### 2. 주요 결과

- **성능 및 효율성:** AROMA는 LoRA $r=8$ 및 AdaLoRA $r=8$ 대비 **학습 가능 파라미터를 10% 미만으로 사용하면서도 더 우수한 성능**을 보였다. 특히 GLUE의 CoLA, MRPC, RTE, SST-2 태스크에서 다른 베이스라인을 능가했다.
- **유효 랭크(Effective Rank):** LoRA와 AdaLoRA는 할당된 랭크 대비 실제 기여하는 유효 랭크 비율이 낮은 반면(AdaLoRA의 경우 약 50%), AROMA는 **90% 이상의 매우 높은 유효 랭크 비율**을 보였다.
- **연산 속도:** SVD 계산이 필요 없고 랭크-1 단위로 학습하므로 시간 효율성이 매우 높다. GLUE 벤치마크 평균 에폭당 학습 시간이 LoRA의 76.1%, AdaLoRA의 28.5% 수준으로 단축되었다.
- **LLaMA3-8B 결과:** AROMA $r=1$ 설정만으로도 많은 베이스라인을 앞질렀으며, $r=8$ 설정에서는 대부분의 벤치마크에서 최상위권 성능을 달성했다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **서브스페이스 탐색의 효율성:** AROMA가 높은 유효 랭크 비율을 가지는 이유는 'Reset' 메커니즘 덕분이다. 옵티마이저를 리셋함으로써 각 랭크-1 구성 요소가 서로 중복되지 않는 독립적인 정보 서브스페이스를 학습할 수 있게 된다.
- **레이어별 특성:** 실험 결과, 랭크 분포가 주로 얕은 레이어(shallower layers)와 $W_v, W_o$ 행렬에 집중되는 경향이 발견되었다. 이는 모델의 특정 부분이 태스크 적응에 더 민감하게 반응함을 시사한다.

### 2. 한계 및 비판적 해석

- **범용성 검증 부족:** 본 논문은 NLU와 상식 추론 태스크에 집중하였으나, 최근 중요성이 커진 멀티모달(Multimodal) 애플리케이션에서의 성능은 검증되지 않았다.
- **초거대 모델 확장성:** 1,000억 개 이상의 파라미터를 가진 초거대 모델에서도 동일한 랭크 성장 역학이 작동하는지는 추가적인 확인이 필요하다.
- **하이퍼파라미터 의존성:** $\epsilon_{in}$과 $\epsilon_{out}$이라는 수렴 임계값에 따라 최종 랭크가 결정되므로, 이 값들을 설정하는 과정에서의 민감도 분석이 더 상세히 제공되었다면 좋았을 것이다.

## 📌 TL;DR

AROMA는 LoRA의 정적 랭크 할당과 AdaLoRA의 높은 연산 비용 및 설정 민감도 문제를 해결하기 위해, **0에서부터 필요한 랭크를 점진적으로 키워가는 '자율적 랭크 성장' 프레임워크**를 제안한다. 듀얼 루프 수렴 구조와 옵티마이저 리셋 전략을 통해 매우 적은 파라미터만으로도 각 레이어에 최적화된 랭크를 찾아내며, 결과적으로 기존 LoRA 계열 방법론보다 **훨씬 적은 파라미터와 연산 시간으로 더 높은 성능과 유효 랭크 효율성**을 달성하였다. 이 연구는 향후 자율적 PEFT 및 지속 학습(Continual Learning) 분야에 중요한 기초가 될 가능성이 높다.
