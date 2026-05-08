# Norm-Bounded Low-Rank Adaptation

Ruigang Wang, Krishnamurthy (Dj) Dvijotham, Ian R. Manchester (2025)

## 🧩 Problem to Solve

본 논문은 매개변수 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 과정에서 발생하는 학습 불안정성과 모델 망각 문제를 해결하고자 한다.

기존의 Low-Rank Adaptation(LoRA)은 가중치 업데이트 행렬의 Rank(계수)를 제한함으로써 학습 가능한 매개변수 수를 줄이는 데 집중하였다. 그러나 행렬의 Rank가 낮더라도 Schatten p-norm(예: Nuclear, Frobenius, Spectral norm)으로 대표되는 행렬의 Norm(크기)은 매우 커질 수 있다. 저자들은 이러한 Norm의 무제한적인 증가가 다음과 같은 심각한 문제를 야기한다고 주장한다:

1. **학습 불안정성 및 하이퍼파라미터 민감도**: 가중치 Norm이 과도하게 커지면 학습률(Learning Rate)이나 학습 에폭(Epoch) 수와 같은 하이퍼파라미터 변화에 모델 성능이 매우 민감하게 반응하며, 이는 학습의 불안정성으로 이어진다.
2. **치명적 망각(Catastrophic Forgetting)**: 미세 조정 과정에서 업데이트 행렬의 Norm이 커질수록 사전 학습된(Pre-trained) 모델이 원래 가지고 있던 일반화 성능이 급격히 저하되는 현상이 발생한다.

따라서 본 논문의 목표는 Rank뿐만 아니라 행렬의 Norm을 명시적으로 제어할 수 있는 새로운 파라미터화 방법을 제안하여, 학습 효율성을 높이고 모델 망각을 방지하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **NB-LoRA(Norm-Bounded Low-Rank Adaptation)**라는 새로운 행렬 파라미터화 기법을 제안한 것이다.

핵심 아이디어는 가중치 업데이트 행렬 $W$의 각 특이값(Singular Value)에 명시적인 상한선(Bound)을 두는 것이다. 이를 통해 사용자가 지정한 임의의 Schatten p-norm 제약 조건을 만족시키면서도, 최적화 과정에서는 제약 조건이 없는(Unconstrained) 자유 변수를 사용하여 학습할 수 있는 매끄럽고 완전한(Smooth and Complete) 매핑 함수를 구축한 점이 가장 큰 설계적 특징이다.

## 📎 Related Works

논문에서는 LoRA의 한계를 극복하려는 여러 기존 연구를 소개한다.

- **정규화 기반 접근법**: 일부 연구에서는 사전 학습 모델과 미세 조정 모델 간의 유클리드 거리를 보존하거나, DoRA와 같이 가중치를 분해하여 크기를 조정하는 방식을 제안하였다.
- **SVD 기반 접근법**: PiSSA나 AdaLoRA와 같이 SVD(특이값 분해)를 통해 초기값을 설정하거나 Rank를 동적으로 할당하는 방식이 제안되었다.

그러나 저자들은 이러한 기존 방식들이 특이값에 대한 명시적인 상한선(Explicit Norm Bound)을 설정하여 제어하지 않는다는 점을 한계로 지적하며, NB-LoRA와의 차별성을 강조한다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

NB-LoRA는 표준 LoRA와 유사한 구조를 가지지만, 가중치 업데이트 행렬 $W$를 생성하는 방식이 다르다. 학습 가능한 자유 변수 $\tilde{A} \in \mathbb{R}^{r \times m}$와 $\tilde{B} \in \mathbb{R}^{r \times n}$를 입력으로 받아, 특정한 변환을 거쳐 $W$를 도출한다.

### 주요 구성 요소 및 방정식

1. **Cayley Transformation**:
   자유 변수 $\tilde{A}, \tilde{B}$로부터 반직교(Semi-orthogonal) 행렬을 생성하기 위해 Cayley 변환을 사용한다.
   $$\begin{bmatrix} A \\ B \end{bmatrix} = \text{Cayley}\left(\begin{bmatrix} \tilde{A}^T \\ \tilde{B}^T \end{bmatrix}\right)$$
   여기서 $\text{Cayley}(\begin{bmatrix} X \\ Y \end{bmatrix}) = \begin{bmatrix} (I-Z)(I+Z)^{-1} \\ -2Y(I+Z)^{-1} \end{bmatrix}$ 이며, $Z = X - X^T + Y^T Y$로 정의된다.

2. **Weight Adaptation Matrix 생성**:
   최종 업데이트 행렬 $W$는 다음과 같이 계산된다.
   $$W = 2 A^T S B$$
   여기서 $S = \text{diag}(s)$는 특이값의 상한선을 결정하는 대각 행렬이다. 이 구조를 통해 $W$의 모든 특이값 $\sigma_j(W)$는 $S$의 대각 성분 $s_j$보다 작거나 같음($\sigma_j(W) \le s_j$)이 보장된다.

3. **Schatten p-norm 제약 구현**:
   사용자가 설정한 Norm 상한 $\delta$를 만족시키기 위해 $s$를 다음과 같이 파라미터화한다.
   $$s = \delta \left( \text{Softmax}(\frac{1}{\eta} v) \right)^{1/p}$$
   여기서 $v$는 학습 가능한 벡터이며, 이 식을 통해 $W$의 Schatten p-norm $\|W\|_{S_p} \le \delta$가 강제된다.

### 학습 절차 및 특성

- **초기화**: 표준 LoRA와 마찬가지로 $\tilde{A}=0$으로 설정하여 초기 $W=0$에서 시작할 수 있게 설계되었다.
- **그라디언트 분석**: NB-LoRA는 파라미터 구조상 초기 그라디언트가 너무 작아지는 문제를 방지하며, Norm 상한으로 인해 그라디언트 폭주가 억제되어 학습 안정성이 높다.

## 📊 Results

### LLM 미세 조정 실험

- **데이터셋 및 모델**: LLaMA-2(7B, 13B), LLaMA-3(8B), Mistral-7B 모델을 사용하였으며, GSM8K, MATH, HumanEval, MBPP 등의 벤치마크에서 평가하였다.
- **비교 대상**: LoRA, DoRA, PiSSA.
- **정량적 결과**:
  - NB-LoRA는 거의 모든 태스크에서 기존 방법들보다 우수한 성능을 보였다.
  - 특히 **Rank 예산이 매우 낮을 때(r=16)**, 다른 방법들은 성능이 급격히 하락하는 반면, NB-LoRA는 $r=128$일 때와 유사한 높은 성능을 유지하는 강건함을 보였다.
  - 학습 속도 측면에서도 더 적은 스텝 만에 더 낮은 손실 함수(Loss) 값에 도달하였다.

### Vision Transformer(ViT) 실험

- **작업**: ImageNet-21k로 사전 학습된 ViT-B/16 모델을 CIFAR-100과 Food-101로 미세 조정.
- **모델 망각 측정**: 미세 조정 후 원래 데이터셋인 ImageNet-1k에서의 정확도를 측정하여 '망각' 정도를 평가하였다.
- **결과**:
  - **Adaptation vs Forgetting**: NB-LoRA는 타겟 데이터셋(CIFAR-100)에서 높은 성능을 달성하면서도, 소스 데이터셋(ImageNet-1k)의 성능 하락을 최소화하여 가장 효율적인 트레이드-오프를 보여주었다.
  - **하이퍼파라미터 강건성**: 학습률, Rank, 학습 에폭 변화에 대해 LoRA는 성능 변동이 매우 심했으나, NB-LoRA는 매우 일관된 성능을 유지하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

NB-LoRA의 성공 요인은 Rank와 Norm을 동시에 제어함으로써 최적화 경로를 효율적으로 제한한 데 있다. 특히 저차원(Low-rank) 설정에서 NB-LoRA가 강한 이유는, 제한된 Rank 내에서 Norm의 힘을 효율적으로 집중시켜 최적의 로컬 옵티멈으로 빠르게 이동시키면서도, 상한선(Bound)이 파라미터의 과도한 변화를 막아 안정성을 제공하기 때문이다.

### 한계 및 비판적 해석

1. **계산 비용**: Cayley 변환 과정에서 $r \times r$ 크기의 행렬 역행렬 계산이 필요하다. 비록 $r$이 작아 실제 오버헤드는 적지만, 이론적으로는 추가 연산이 발생한다.
2. **대리 지표(Proxy)의 한계**: 본 논문에서 사용한 Norm 제약은 모델 망각을 방지하기 위한 편리한 '대리 지표'일 뿐, 망각 현상 자체를 직접적으로 최적화하는 공식은 아니다. 따라서 향후 연구에서는 Norm 상한을 학습 과정 중에 동적으로 조정하는 방식 등이 고려될 필요가 있다.

## 📌 TL;DR

NB-LoRA는 PEFT에서 가중치 업데이트 행렬의 **Rank와 Schatten p-norm을 동시에 명시적으로 제어**하는 새로운 파라미터화 방법을 제안한다. 이를 통해 기존 LoRA가 겪던 학습 불안정성과 하이퍼파라미터 민감도 문제를 해결하였으며, 특히 **매우 낮은 Rank에서도 높은 성능을 유지**하고 **사전 학습 모델의 지식을 보존(망각 방지)**하는 데 탁월한 효과가 있음을 입증하였다. 이 연구는 향후 초소형 어댑터를 이용한 효율적인 모델 전이 학습 연구에 중요한 기반이 될 것으로 보인다.
