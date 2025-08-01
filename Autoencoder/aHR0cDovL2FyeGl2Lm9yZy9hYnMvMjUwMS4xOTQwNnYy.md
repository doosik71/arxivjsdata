# Low-Rank Adapting Models for Sparse Autoencoders
Matthew Chen, Joshua Engels, Max Tegmark

## Problem to Solve
언어 모델(LM)의 표현을 희소한 선형 잠재 벡터로 분해하는 **희소 오토인코더(SAE)**는 모델 해석 가능성을 높이는 데 유용합니다. 그러나 기존 SAE는 두 가지 주요 한계점을 가집니다. 첫째, SAE 재구성 결과를 모델에 삽입하여 순전파를 수행할 때 교차 엔트로피 손실(cross entropy loss, L_SAE)이 원래 모델(L_BASE)에 비해 **크게 증가**하여 성능이 저하됩니다. 둘째, 언어 모델 기울기(gradient)를 사용하는 최신 SAE 개선 기법(`e2e SAE` 등)은 학습 중 **많은 계산 비용이 드는 역전파(backward pass)**를 필요로 합니다. 이 논문은 이러한 한계점을 극복하고 해석 가능성과 성능 사이의 파레토 최적화를 개선하는 것을 목표로 합니다.

## Key Contributions
*   **모델 자체를 SAE에 맞게 최적화**: 기존 SAE를 중심으로 언어 모델 자체를 개선하는 데 초점을 맞춘 **최초의 연구**입니다.
*   **교차 엔트로피 손실 감소**: `Gemma Scope` 계열 SAE를 사용하여 `L_SAE - L_BASE` 격차를 **30%에서 55%까지 감소**시킵니다 (최종 값 0.01에서 0.17 nats). 특히 낮은 희소성(sparsity)과 대규모 모델에서 큰 개선 효과를 보입니다.
*   **학습 속도 향상**: 기존 `e2e SAE` 대비 동일한 다운스트림 교차 엔트로피 손실을 달성하는 데 `Gemma-2-2B`에서 **3배에서 20배 빠르며**, `Llama-3.2-1B`에서 **2배에서 10배 빠릅니다**. 또한 언어 모델 역전파 횟수를 `Gemma-2-2B`에서 130배, `Llama-3.2-1B`에서 40배 줄입니다.
*   **다중 SAE 적응 가능성**: 여러 SAE를 동시에 삽입할 때 `L_SAE`를 크게 줄여 회로 분석(circuit analysis)에 대한 잠재력을 보여줍니다.
*   **다운스트림 성능 개선**: `SAEBench` 및 새로운 `Feature Steering` 지표에서 정량적 개선을 입증합니다. 모델의 전반적인 언어 능력에 해를 끼치지 않습니다.
*   **효율성 메커니즘 분석**: SAE 다음 레이어에 어댑터를 학습하는 것만으로 전체 레이어에 LoRA를 적용하는 손실 감소의 **88.14%**를 달성할 수 있음을 보여줍니다. 또한 LoRA 적용 시 활성화 함수(activations)와 원래 모델 활성화 간의 거리가 감소하고 코사인 유사도가 증가하여 계산 경로가 더 유사해짐을 확인합니다.

## Methodology
본 연구는 **저랭크 적응(Low-Rank Adapters, LoRA)**을 사용하여 사전 학습된 SAE를 중심으로 언어 모델 자체를 미세 조정하는 근본적으로 다른 접근 방식을 사용합니다.

1.  **SAE 삽입**: **사전 학습된 고정된(frozen) SAE**를 특정 레이어 $\ell$ 직후에 삽입합니다. 이 SAE는 원래 활성화 $x_{\ell}$를 재구성된 활성화 $\hat{x}_{\ell} = \text{SAE}(x_{\ell})$로 변환합니다.
2.  **재구성된 활성화 전파**: 재구성된 활성화 $\hat{x}_{\ell}$는 나머지 레이어($\ell+1$부터 $L$까지)를 통해 전파되어 최종 출력 $\hat{y}$를 생성합니다.
3.  **LoRA 어댑터 추가**: LoRA 어댑터는 언어 모델의 각 **MLP 및 어텐션 서브레이어**에 추가됩니다. 원래 가중치 행렬 $W_0 \in \mathbb{R}^{d \times k}$에 대해 저랭크 행렬 $A \in \mathbb{R}^{d \times r}$와 $B \in \mathbb{R}^{r \times k}$ ($r \ll \min(d,k)$)를 추가하여 순전파를 $\hat{h}(x) = W_0 x + ABx$로 수정합니다.
    *   `JumpReLU SAE`의 경우 평균 희소성이 영향을 받지 않도록 SAE **이후의 레이어만** 미세 조정합니다.
    *   `TopK SAE`의 경우 학습 중 `TopK` 제약 조건을 유지하여 **모든 레이어**에 어댑터를 학습할 수 있습니다.
4.  **LoRA 매개변수 학습**: LoRA 어댑터의 매개변수 $\Theta = \{A_i\} \cup \{B_i\}$만 학습합니다. 원래 언어 모델과 SAE는 모두 고정됩니다.
5.  **학습 목표**: SAE가 삽입된 모델의 다음 토큰 확률 분포와 원래 모델의 다음 토큰 확률 분포 사이의 **KL 발산($\text{KL}(\hat{y},y)$)을 최소화**합니다. 이는 SAE 강화 모델을 원래 모델의 동작에 맞춰 정렬하는 것을 목표로 합니다.

## Results
*   **L_SAE 개선**: 다양한 SAE 희소성, 폭, 언어 모델 크기, LoRA 랭크 및 모델 레이어에 걸쳐 `L_SAE - L_BASE` 간격을 **최소 30%에서 최대 55%까지 감소**시켰습니다. LoRA 랭크가 클수록 최종 `L_SAE`는 더 낮아지는 경향을 보였습니다. 특히 대규모 모델에서 본 방법의 효과가 증가했습니다.
*   **계산 효율성**: `Gemma-2-2B` 및 `Llama-3.2-1B`에서 `e2e SAE` 대비 동일한 `L_SAE` 달성 시 **2배에서 20배 빠른 실제 시간(wall clock time)**을 보였습니다. 역전파 횟수는 `Gemma-2-2B`에서 130배, `Llama-3.2-1B`에서 40배 적었습니다.
*   **다중 SAE 적용**: `Llama-3.1-8B`에 여러 `Llama Scope SAE`를 동시에 삽입했을 때, 예를 들어 7개의 SAE에 대한 복합 CE 손실(compound CE loss)이 7.83 nats에서 2.78 nats로 크게 감소했습니다.
*   **다운스트림 지표 향상**: `SAEBench`에서 `SCR`, `TPP`, `SPARSEPROBING` 지표를 개선했습니다. `Feature Steering` 실험을 통해 LoRA 모델이 SAE 잠재 벡터를 사용하여 특정 동작을 유도하거나 억제하는 데 더 효과적임을 입증했습니다.
*   **일반 언어 모델 능력 유지**: `MMLU`, `HellaSwag`, `TruthfulQA` 벤치마크에서 SAE가 삽입된 LoRA 모델의 성능이 원래 모델과 비교했을 때 손상되지 않았으며, 오히려 더 나은 결과를 보여 일반 언어 모델 능력에 해를 끼치지 않음을 확인했습니다.
*   **내부 메커니즘 분석**: 토큰별 손실 개선을 분석한 결과, 전반적인 개선은 대부분의 토큰에서 작은 손실 감소(약 $10^{-2}$ ~ 1 nats)로부터 오는 것으로 나타났습니다. LoRA 적용 시 활성화 간의 `L2` 거리가 약간 감소하고 코사인 유사도가 증가하여, 어댑터가 적용된 모델이 원래 모델의 계산 경로를 더 가깝게 따른다는 것을 보여주었습니다. 또한 SAE 레이어 바로 다음 레이어에만 LoRA 어댑터를 학습하는 것이 전체 레이어에 학습하는 것과 유사한 손실 감소(88.14%)를 달성하여 손실 개선 메커니즘이 비교적 간단할 수 있음을 시사했습니다.