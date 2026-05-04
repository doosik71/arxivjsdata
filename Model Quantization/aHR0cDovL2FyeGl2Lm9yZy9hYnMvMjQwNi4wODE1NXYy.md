# QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts

Pingzhi Li, Xiaolong Jin, Zhen Tan, Yu Cheng, Tianlong Chen (2024)

## 🧩 Problem to Solve

본 논문은 Mixture-of-Experts (MoE) 구조를 가진 거대 언어 모델(LLM)의 막대한 메모리 오버헤드 문제를 해결하기 위한 Post-Training Quantization (PTQ) 전략을 연구한다. MoE 모델은 추론 시 Sparse Activation을 통해 계산량(FLOPs)을 일정하게 유지하면서 파라미터 수를 확장할 수 있다는 장점이 있으나, 전체 파라미터 크기가 매우 커서 메모리 소모가 극심하다.

기존의 모델 압축 기법인 전문가 병합(Expert Merging)이나 가지치기(Pruning)는 막대한 재학습 비용이 발생하거나 특정 작업(Task-specific)에 국한되는 한계가 있다. 반면 PTQ는 재학습 없이 가중치 정밀도를 낮출 수 있는 효율적인 대안이다. 그러나 기존의 PTQ 방법론들은 모델 전체에 동일한 비트 정밀도를 적용하는 Uniform Quantization 방식을 사용하며, 이는 MoE 고유의 희소 구조(Sparse Structure)와 전문가별 활성화 패턴의 차이를 반영하지 못해 성능 저하를 초래한다. 따라서 본 논문의 목표는 MoE 구조의 특성을 고려한 세밀한 정밀도 설정(Fine-grained Precision Setup)을 탐구하고, 이를 최적화하는 데이터 기반의 비트 할당 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 MoE 모델 내의 구성 요소마다 중요도가 다르므로, 구조적 특성에 따라 정밀도를 다르게 할당하는 Mixed-Precision Quantization이 필요하다는 것이다. 주요 기여 사항은 다음과 같다.

1.  **QuantMoE-Bench 구축**: MoE 모델의 PTQ를 체계적으로 분석하기 위한 최초의 벤치마크를 제안한다. Mixtral-8x7B 및 DeepSeek-MoE-16B-base 모델과 6가지 벤치마크 작업을 통해 정밀도 할당 전략을 평가한다.
2.  **MoE 양자화 원리 규명**: 실험을 통해 Attention 레이어가 FFNN보다 더 높은 정밀도를 필요로 하며, Shared Expert가 Token-conditioned Expert보다, 그리고 모델의 앞부분(Early layers)이 뒷부분보다 더 높은 정밀도를 유지해야 성능 저하가 적다는 원리를 발견하였다.
3.  **데이터 기반 비트 할당 기술 제안**: 가중치의 이상치(Outlier)를 감지하여 정밀도를 결정하는 Outlier-aware Linear Layer Scorer와, 블록별 활성화 패턴을 통해 중요도를 예측하는 MoE Block Importance Predictor를 도입하였다.
4.  **SOTA 성능 달성**: 제안한 세밀한 Mixed-precision 기법을 통해 기존 GPTQ(Uniform) 대비 평균 성능을 64.30%에서 65.35%로 향상시켰다.

## 📎 Related Works

**Mixture-of-Experts (MoE)**: 라우터 네트워크를 통해 입력 토큰에 따라 특정 전문가(Expert)만을 선택적으로 활성화함으로써 모델 용량을 확장하는 구조이다. Mixtral이나 DeepSeek-MoE 등이 대표적이며, 계산 효율성은 높으나 메모리 요구량이 매우 크다는 단점이 있다.

**MoE Compression**: 기존 연구들은 주로 전문가의 수를 줄이는 방식에 집중하였다. 특정 작업에 중요하지 않은 전문가를 제거하는 Pruning이나, 유사한 전문가를 병합하는 Merging 방식이 있으나, 이들은 대부분 Task-specific fine-tuning을 전제로 하기에 범용적인 LLM 적용에 한계가 있다.

**Post-Training Quantization (PTQ)**: GPTQ, SmoothQuant, AWQ 등은 고정밀도 가중치를 저정밀도로 변환하여 메모리를 절감한다. 특히 GPTQ는 Hessian 정보를 이용하여 재구성 오차를 최소화하는 방식을 사용한다. 하지만 이러한 기법들은 Dense 모델 중심으로 연구되었으며, MoE의 희소 활성화 특성에 최적화된 분석은 부족한 실정이다.

## 🛠️ Methodology

### 1. 기본 양자화 프레임워크
본 논문은 가중치 전용 양자화 방법인 GPTQ를 기본 엔진으로 사용한다. GPTQ는 각 레이어에 대해 다음과 같은 재구성 문제(Reconstruction problem)를 해결함으로써 최적의 양자화 가중치 $\hat{W}$를 찾는다.

$$\text{argmin}_{\hat{W}} \|WX - \hat{W}X\|_2^2$$

여기서 $W$는 원래 가중치, $X$는 캘리브레이션 데이터에서 얻은 입력값이다.

### 2. MoE 구조적 특성에 따른 비트 할당 (Heuristics)
논문은 Pareto-Optimal Bit Allocation 관점에서 성능과 메모리 사용량의 트레이드오프를 분석하며, 다음과 같은 구조적 가이드라인을 제시한다.

*   **Attention vs FFNN**: Attention 레이어는 모든 토큰에 대해 활성화되므로 FFNN보다 더 높은 정밀도(예: 4-bit)를 우선적으로 할당한다.
*   **Shared Experts**: 모든 토큰이 공유하는 Shared Expert는 일반 전문가보다 중요도가 높으므로 더 많은 비트를 할당한다.
*   **Expert Usage Frequency**: 캘리브레이션 데이터에서 라우팅 빈도가 높은 전문가에게 더 높은 정밀도를 부여한다.
*   **Layer Position**: 모델의 앞부분에 위치한 MoE 블록들이 뒷부분보다 성능에 더 큰 영향을 미치므로 우선적으로 높은 정밀도를 할당한다.

### 3. 데이터 기반 최적화 기법

#### (1) Outlier-Aware Linear Layer Scorer
가중치 분포에서 극단적인 값(Outlier)이 많을수록 양자화 오차가 커진다는 점에 착안하여, 각 선형 레이어의 이상치 점수를 계산한다. 가중치 행렬 $W \in \mathbb{R}^{m \times n}$에 대해, 각 열의 최대 절대값과 평균 절대값의 비율 중 최대값을 점수로 정의한다.

$$\text{outlier-score}(W) = \max_{j} \left( \frac{\max(|W_{:,j}|)}{\text{mean}(|W_{:,j}|)} \right)$$

이 점수가 높은 레이어일수록 양자화가 어렵다고 판단하여 더 높은 비트를 할당한다.

#### (2) MoE Block Importance Score Predictor
MoE 블록의 입력 텐서 $x$와 출력 텐서 $y$ 사이의 코사인 유사도가 높을수록 해당 블록의 연산 결과가 입력과 크게 다르지 않아 중요도가 낮다고 판단한다. 이를 위해 경량화된 2층 FFNN 예측기(BSP)를 학습시켜 각 블록의 중요도를 예측하고, 예측 점수가 낮은(즉, 유사도가 높은) 블록에 낮은 정밀도를 할당한다.

## 📊 Results

### 실험 설정
*   **대상 모델**: Mixtral-8x7B, DeepSeek-MoE-16B-base
*   **평가 작업**: WinoGrande, COPA, OBQA, HellaSwag, PIQA, MMLU (6개 작업)
*   **비교 대상**: Uniform GPTQ (2-bit, 3-bit, 4-bit) 및 제안하는 Mixed-precision 전략
*   **지표**: 각 작업별 정답률(%) 및 평균 성능

### 주요 결과
1.  **Mixed-Precision의 우위**: Figure 3의 Pareto-frontier 분석 결과, 동일한 평균 비트 예산 하에서 제안한 Mixed-precision 방식이 Uniform 방식보다 월등히 높은 성능을 보였다.
2.  **구조적 통찰의 검증**:
    *   Attention 레이어에 4-bit를 할당했을 때 FFNN에 할당했을 때보다 성능 향상 폭이 훨씬 컸다 (Figure 4).
    *   Shared Expert에 높은 정밀도를 부여하는 것이 무작위 전문가에게 부여하는 것보다 효과적이었다.
    *   첫 4~8개 블록을 고정밀도로 유지하는 것이 마지막 블록들을 유지하는 것보다 성능 유지에 유리했다.
3.  **정량적 성능 향상**: DeepSeek-MoE-16B-base 모델에서 `+Attn +Shared +Freq +FirstL` 조합을 적용했을 때, baseline GPTQ 대비 평균 성능이 $64.30\% \rightarrow 65.35\%$로 약 $1.05\%$p 상승하였다.
4.  **데이터 기반 스코어러의 효과**: Outlier-aware scorer와 Block importance predictor를 적용했을 때, 무작위 선택 기반의 Mixed-precision보다 추가적인 성능 향상(약 $0.97\%$)이 관찰되었다.

## 🧠 Insights & Discussion

본 연구는 MoE 모델의 양자화가 단순히 전체 비트를 낮추는 문제가 아니라, **"어디에 정밀도를 집중할 것인가"**의 문제임을 입증하였다. 

**강점 및 발견**:
*   **활성화 빈도와 정밀도의 상관관계**: 모든 토큰이 거치는 경로(Attention, Shared Experts, Early Layers)는 정보의 병목 지점이 되므로 높은 정밀도가 필수적이다.
*   **구조적 비대칭성 활용**: MoE의 Sparse Routing 특성으로 인해 발생하는 가중치 활용의 불균형을 정밀도 할당의 근거로 활용하여 메모리 효율과 성능을 동시에 잡았다.

**한계 및 논의**:
*   **캘리브레이션 데이터 의존성**: Expert usage frequency나 Block predictor가 캘리브레이션 데이터의 분포에 의존하므로, 데이터셋의 대표성에 따라 최적의 비트 할당이 달라질 가능성이 있다.
*   **연산 오버헤드**: Mixed-precision은 메모리를 절감하지만, 하드웨어 수준에서 서로 다른 비트 정밀도의 연산을 효율적으로 처리하기 위한 커널 최적화가 추가로 필요하다.

## 📌 TL;DR

본 논문은 MoE 모델의 메모리 문제를 해결하기 위해 구조적 특성을 반영한 **Mixed-Precision Post-Training Quantization** 전략을 제안한다. **QuantMoE-Bench**를 통해 Attention 레이어, Shared Expert, 모델 전반부 레이어가 양자화에 더 민감하다는 것을 밝혀냈으며, 이를 최적화하기 위한 **Outlier-aware Scorer**와 **Block Importance Predictor**를 도입하여 SOTA 수준의 압축 성능을 달성하였다. 이 연구는 향후 초거대 MoE 모델의 효율적인 배포와 메모리 최적화를 위한 핵심적인 가이드라인을 제공한다.