# MAMBAPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba

Masakazu Yoshimura, Teruaki Hayashi & Yota Maeda (2025)

## 🧩 Problem to Solve

본 논문은 최근 Transformer의 대안으로 주목받고 있는 State Space Model (SSM) 기반의 Mamba 아키텍처를 다운스트림 태스크에 효율적으로 적응시키기 위한 Parameter-Efficient Fine-Tuning (PEFT) 방법론을 탐구한다.

Transformer 기반 모델들은 방대한 데이터와 파라미터를 통해 강력한 성능을 보여주지만, 시퀀스 길이에 따라 연산 복잡도가 제곱으로 증가하는 효율성 문제가 있다. Mamba는 이를 해결하기 위해 선형 시간 복잡도를 가지는 SSM을 도입하여 긴 시퀀스를 효율적으로 처리하며, 유사한 파라미터 규모의 Transformer보다 NLP 및 CV 작업에서 우수한 성능을 보인다. 그러나 Transformer 분야에서 매우 활발하게 연구된 PEFT 기술이 Mamba 아키텍처에서는 어떻게 작동하는지, 그리고 Mamba의 구조적 특성에 최적화된 PEFT 방법은 무엇인지에 대한 체계적인 분석과 벤치마크가 부족한 상황이다. 따라서 본 연구의 목표는 기존 PEFT 방법론을 Mamba에 적용해보고, Mamba 전용 PEFT 방법을 제안하며, 최적의 PEFT 조합을 찾는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba를 위한 PEFT의 포괄적인 탐색과 벤치마킹에 있으며, 구체적인 설계 아이디어는 다음과 같다.

1.  **Mamba 전용 PEFT 방법론 제안**: Mamba의 선택적 스캔(Selective Scan) 구조와 상태 공간(State Space) 특성을 활용한 **Additional-scan**과 **Affix-tuning**을 제안하였다.
2.  **기존 PEFT의 재설계**: Transformer용 LoRA와 Prompt-tuning을 Mamba의 특성에 맞게 수정하여 **Partial LoRA**와 위치 최적화된 Prompt-tuning을 제시하였다.
3.  **하이브리드 PEFT 검색 프레임워크**: 여러 PEFT 방법을 단순히 결합하는 것이 아니라, 2단계 검색 기법(조합 탐색 $\rightarrow$ 하이퍼파라미터 최적화)을 통해 최적의 조합을 찾는 효율적인 프레임워크를 제안하였다.
4.  **Mamba-Transformer PEFT 비교 분석**: 실험을 통해 Mamba가 Transformer보다 PEFT에 더 효과적으로 반응하며, 파라미터 증가에 따른 성능 향상 폭이 더 크고 오버피팅에 더 강건하다는 사실을 발견하였다.

## 📎 Related Works

### Mamba 및 SSM
SSM은 입력 $X_t$를 은닉 상태 $H_t$로 변환하여 출력 $Y_t$를 생성하는 시스템으로, 기본적으로 다음과 같은 연속 시간 방정식을 기반으로 한다.
$$H_t = AH_{t-1} + BX_t, \quad Y_t = CH_t + DX_t$$
Mamba는 이를 이산화하고, $A, B, C, D$ 파라미터를 데이터에 따라 동적으로 결정하는 **Selective State Space Model (S6)** 구조를 채택하여 Transformer의 Attention 메커니즘과 유사한 능력을 가지면서도 선형 시간 복잡도를 달성하였다.

### Transformer PEFT
Transformer의 PEFT는 크게 네 가지 범주로 나뉜다.
- **Partial-tuning**: 모델의 일부 파라미터(예: Bias)만 업데이트하는 방식 (BitFit 등).
- **Additive methods**: 작은 어댑터 모듈이나 소프트 토큰을 추가하는 방식 (Adapter, Prompt-tuning).
- **Reparameterization**: 가중치를 저차원 행렬의 합으로 표현하는 방식 (LoRA).
- **Hybrid**: 위의 방법들을 혼합하여 사용하는 방식.

기존 연구들은 주로 Transformer의 Attention 층에 집중되어 있으며, Mamba에 LoRA를 단순히 적용한 일부 연구가 존재하지만, Mamba의 특수한 구조(Selective Scan, Conv1d 등)를 고려한 깊이 있는 탐구는 부족했다.

## 🛠️ Methodology

본 논문은 PEFT 방법론을 세 가지 단계(단순 적용 $\rightarrow$ 재설계 $\rightarrow$ 신규 제안)로 나누어 접근한다.

### 1. 단순 적용 (Simple Adaptation)
- **ParallelAdapter**: Transformer의 FFN에 적용하던 방식을 Mamba의 `out_proj` 층에 병렬로 부착하여 적용한다.
- **LoRA**: Mamba 내의 다양한 Linear 층(`embedding`, `in_proj`, `x_proj`, `dt_proj`, `out_proj`)에 각각 적용하여 효과를 측정한다.

### 2. 재설계 및 개선 (Re-designing)
- **Partial LoRA**: Mamba의 중간 특징들($X, Z, dt, B, C$)이 서로 다른 성질을 가진다는 점에 착안하여, 전체 가중치가 아닌 특정 출력 특징에 해당하는 부분에만 LoRA를 적용하는 $\text{LoRA}_p$를 제안한다.
- **Prompt-tuning**: Mamba는 시계열 모델이므로 토큰의 위치가 중요하다. 이에 따라 프롬프트 토큰을 입력의 시작(Prefix), 중간(Infix), 끝(Suffix)에 삽입하는 세 가지 유형을 실험한다.
- **Affix-tuning**: Transformer의 Prefix-tuning은 Attention 메커니즘 기반이므로 Mamba에 직접 적용하기 어렵다. 이를 해결하기 위해 SSM 입력 전단에 소프트 토큰을 추가하고, SSM 출력 후 해당 위치의 토큰을 제거(discard)함으로써 원래 입력 시퀀스의 위치 관계를 유지하는 방식을 제안한다.

### 3. Mamba 전용 신규 방법 (New PEFT Methods)
- **Partial-tuning (Bias, A, D, Conv1d tuning)**: Mamba는 Linear 층에 Bias가 없는 경우가 많다. 대신 $A, D$ 파라미터와 Causal Conv1d의 가중치를 튜닝한다. 또한, 사전 학습된 모델을 보존하기 위해 일반적인 Weight Decay 대신 $|W - W_{\text{pretrain}}|^2$ 형태의 가중치 감쇠를 적용한다.
- **Additional-scan**: SSM의 상태 차원(State Dimension) $N$을 $N+N'$로 확장하여 새로운 정보를 저장하도록 한다.
  $$H^{[L, N+N']}_t = A^{[L, N+N']}_t \circ H^{[L, N+N']}_{t-1} + B^{[L, N+N']}_t \circ X^{[D, 1]}_t$$
  이 방법은 기존의 은닉 상태 $h_{t, 1}, \dots, h_{t, N}$에 영향을 주지 않고 추가적인 상태 차원에서만 학습이 이루어지므로 사전 학습된 메모리를 보존하면서 새로운 태스크에 적응할 수 있다. 초기화는 인접한 사전 학습된 $A$ 값의 값을 사용하는 방식을 제안한다.

### 4. 하이브리드 PEFT 검색 (Hybrid PEFT Search)
다양한 PEFT 조합의 탐색 공간이 너무 넓기 때문에 2단계 접근법을 사용한다.
- **Step 1**: 최소 파라미터 설정에서 각 PEFT 방법의 활성화 여부(Boolean)만을 결정하여 최적의 조합을 찾는다.
- **Step 2**: 선택된 조합을 바탕으로 각 방법의 하이퍼파라미터(Rank, Learning rate 등)를 그리디하게 검색하며, 필요 없는 방법은 제거하여 파라미터 효율성을 최적화한다.

## 📊 Results

### 실험 설정
- **모델**: Vim-S (Vision Mamba) 및 Pythia (Language Model) 기반.
- **데이터셋**: 이미지 작업은 VTAB-1k, 언어 작업은 Commonsense Reasoning 태스크 사용.
- **지표**: Test Accuracy 및 학습 시간 비율(Time Ratio), 학습 가능 파라미터 수.

### 주요 결과
1.  **Mamba vs Transformer**: Vim-S에 PEFT를 적용한 결과가 ViT-S에 적용한 것보다 전반적으로 우수했다. 특히 ViT는 파라미터가 증가하면 빠르게 오버피팅되는 경향이 있으나, Vim은 파라미터가 증가해도 성능이 계속 향상되는 경향을 보였다.
2.  **개별 방법론 성능**:
    - **LoRA**: $\text{LoRA}_p(X)$가 매우 효과적이었으며, 특히 작은 차원의 Linear 층에서는 LoRA의 Rank가 Full-rank를 초과해도 성능이 계속 향상되는 '과잉 매개변수화(Over-parameterization)' 효과가 관찰되었다.
    - **Additional-scan**: 적은 파라미터로도 LoRA 및 Affix-tuning과 경쟁 가능한 성능을 보였다.
    - **Affix-tuning**: 특히 모델 규모가 커질수록, 그리고 데이터셋이 클수록 효과가 두드러졌다.
3.  **하이브리드 PEFT**: 단순히 고성능 방법들을 합치는 것보다 제안된 2단계 검색을 통해 최적화된 조합을 사용했을 때 훨씬 적은 파라미터로 더 높은 성능을 달성하였다.
4.  **데이터 규모에 따른 경향**:
    - 데이터가 적을 때 ($\approx 1\text{K}$): $\text{LoRA}_p(X)$가 가장 우수함.
    - 데이터가 많을 때 ($\approx 170\text{K}$): Additional-scan과 Affix-tuning의 효율성이 급격히 증가함.

## 🧠 Insights & Discussion

### 강점 및 발견
- **Mamba의 구조적 이점**: Mamba의 모듈형 구조가 PEFT 추가 파라미터로 인한 사전 학습 메모리의 오염(Corruption)을 방지하여, Transformer보다 PEFT에 더 강건하고 효과적임을 입증하였다.
- **데이터 규모와의 상관관계**: PEFT 방법 선택 시 데이터의 양이 결정적인 요인이 된다는 점을 밝혀냈다. 이는 향후 Mamba 기반 모델을 튜닝할 때 중요한 가이드라인이 된다.

### 한계 및 비판적 해석
- **모델 규모의 확장성**: 본 연구는 Vim-S, Mamba 1.4B 수준에서 진행되었으며, 수십~수백억 개의 파라미터를 가진 초대형 모델에서도 동일한 경향이 나타나는지는 추가 검증이 필요하다.
- **작업 특성**: 일부 시각 작업(예: dSpr-Loc)에서는 여전히 Attention 메커니즘이 Mamba보다 우수한 성능을 보이는데, 이는 Mamba의 순차적 처리 방식이 특정 공간적 위치 파악 작업에는 불리할 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 Mamba 아키텍처를 위한 최초의 포괄적인 PEFT 벤치마크를 수행하고, Mamba의 특성을 반영한 **Additional-scan**, **Affix-tuning**, **Partial LoRA** 등의 방법론을 제안하였다. 실험 결과, Mamba는 Transformer보다 PEFT에 더 효율적이며 오버피팅에 강건하다는 것이 밝혀졌다. 특히 데이터 규모에 따라 최적의 PEFT 방법이 달라짐을 확인하였으며, 제안된 하이브리드 검색 프레임워크를 통해 최적의 PEFT 조합을 찾을 수 있음을 보여주었다. 이 연구는 향후 Mamba 기반의 거대 모델을 효율적으로 배포하고 적응시키는 데 중요한 기초 자료가 될 것이다.