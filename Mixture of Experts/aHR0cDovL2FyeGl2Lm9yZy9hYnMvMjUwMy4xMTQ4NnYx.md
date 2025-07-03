# DeepSeek 모델의 주요 혁신 기술 검토
Chengen Wang, Murat Kantarcioglu

## 해결하고자 하는 문제
DeepSeek-V3 및 DeepSeek-R1은 OpenAI, Anthropic 등 최고 수준의 비공개 소스 모델과 비교할 만한 성능을 달성하면서도 훨씬 적은 훈련 비용을 요구하는 선도적인 오픈 소스 대규모 언어 모델(LLMs)입니다. 이 논문은 DeepSeek 모델의 성공 뒤에 있는 핵심 혁신 기술들을 검토하여 LLM 연구 발전에 기여하고자 합니다. 또한, DeepSeek의 기술 보고서에서 다루지 않았거나 추가 연구가 필요한 공개 문제점과 잠재적인 연구 기회를 제시합니다.

## 주요 기여
*   **Multi-Head Latent Attention (MLA)** 및 **Mixture of Experts (MoE)**를 포함한 트랜스포머 아키텍처 개선 사항을 분석했습니다.
*   훈련 중 **샘플 효율성(sample efficiency)**을 높이는 **Multi-Token Prediction (MTP)** 기술을 검토했습니다.
*   알고리즘, 프레임워크, 하드웨어의 **공동 설계(co-design)** (예: `DualPipe`, `FP8` 혼합 정밀도 훈련)를 통해 훈련 효율성을 크게 향상시킨 점을 다루었습니다.
*   `PPO`의 효율적인 변형인 **Group Relative Policy Optimization (GRPO)** 알고리즘을 설명했습니다.
*   순수 강화 학습 및 지도 미세 조정(SFT)과 강화 학습을 번갈아 사용하는 **반복적 훈련(iterative training)**을 포함한 DeepSeek 모델의 후처리(post-training) 단계를 분석했습니다.
*   DeepSeek 기술 보고서에서 다루지 않거나 추가 연구가 필요한 몇 가지 공개 질문과 잠재적인 연구 방향을 제시했습니다.

## 방법론
이 논문은 DeepSeek 모델의 성공에 기여한 핵심 혁신 기술들을 다음과 같이 분류하고 검토합니다.

*   **트랜스포머 아키텍처 개선:**
    *   **Multi-Head Latent Attention (MLA):** `KV 캐시(cache)` 메모리 사용량을 크게 줄이기 위해 키와 값을 하나의 `잠재 벡터(latent vector)`로 압축하는 **저랭크(low-rank) 키-값(key-value) 공동 압축**을 도입했습니다. 또한, 추론 시 계산 비용을 줄이기 위해 **분리형 로터리 위치 임베딩(Decoupled Rotary Position Embedding, RoPE)**을 사용하여 `RoPE`를 별도의 쿼리와 키로 분리합니다.
    *   **Mixture of Experts (MoE) - DeepSeekMoE:**
        *   **세분화된 전문가 분할(Fine-Grained Expert Segmentation):** 각 `FFN(Feed-Forward Network)`을 더 작은 `m`개의 전문가로 분할하여 활성화되는 전문가들의 조합 유연성을 향상시킵니다.
        *   **공유 전문가 분리(Shared Expert Isolation):** `K_s`개의 전문가를 "공유 전문가"로 지정하여 모든 토큰에 할당함으로써 공통 지식을 포착하고 매개변수 중복을 줄입니다.
        *   **부하 분산(Load Balancing):** `Auxiliary Loss`를 사용하여 전문가 간의 부하를 균형 있게 조절하며, 오버로드된 전문가의 `Top-K` 선택에 영향을 미치는 `바이어스 텀(bias term)` $b_i$를 도입하여 보조 손실 없이 부하를 균형화하는 전략도 사용합니다.

*   **Multi-Token Prediction (MTP):**
    *   각 토큰에 대해 다음 토큰만을 예측하는 대신, `D`개의 추가 토큰을 인과적 사슬(causal chain)로 예측합니다.
    *   훈련 목표는 각 깊이 `k`에서의 `교차 엔트로피 손실(Cross-Entropy Loss)`의 평균인 $L_{MTP} = \frac{\lambda}{D} \sum_{k=1}^D L_k^{MTP}$입니다. 이는 훈련 중 **샘플 효율성**을 높여 성능 향상을 가져오지만, 추가적인 훈련 시간 오버헤드를 발생시킬 수 있습니다.

*   **알고리즘, 프레임워크 및 하드웨어 공동 설계:**
    *   **DualPipe:** 교차 노드 전문가 병렬 처리(cross-node expert parallelism)로 인한 통신 오버헤드를 줄이는 혁신적인 파이프라인 병렬 처리 알고리즘입니다. 순방향 및 역방향 청크 내에서 계산과 통신을 중첩시킵니다.
    *   **FP8 혼합 정밀도 훈련(FP8 Mixed Precision Training):** 대부분의 핵심 `GEMM(General Matrix Multiplication)` 연산을 `FP8` 정밀도로 구현하여 훈련 속도를 높입니다. 임베딩 모듈, 출력 헤드 등 민감한 연산은 원래 정밀도를 유지하여 수치 안정성을 보장합니다.

*   **Group Relative Policy Optimization (GRPO):**
    *   `PPO(Proximal Policy Optimization)`의 효율적인 변형으로, `가치 함수(value function)` 근사를 제거하여 메모리 사용량을 크게 줄입니다.
    *   이전 정책에서 그룹 출력을 샘플링하고 보상 모델에서 얻은 보상을 정규화하여 `어드밴티지(advantage)` $\hat{A}_{i,t}$를 직접 추정합니다.
    *   다음과 같은 목적 함수를 최대화합니다:
        $$J_{GRPO}(\theta) = E[q \sim P(Q),\{o_i\}^G_{i=1} \sim \pi_{\theta_{old}}(O|q)] \frac{1}{G} \sum^G_{i=1} \frac{1}{|o_i|} \sum^{|o_i|}_{t=1} \left( \min \left[ \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})} \hat{A}_{i,t},\text{clip}\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})},1-\epsilon,1+\epsilon \right) \hat{A}_{i,t} \right] -\beta D_{KL}(\pi_\theta||\pi_{ref}) \right)$$

*   **후처리: 베이스 모델에 대한 강화 학습:**
    *   **순수 강화 학습(DeepSeek-R1-Zero):** `DeepSeek-V3-Base` 모델을 SFT 데이터 없이 순수 `GRPO` 강화 학습으로 훈련합니다. 정확도 보상과 형식 보상(예: `<think>` 태그 사용)을 포함하는 보상 함수를 사용합니다.
    *   **콜드 스타트가 포함된 강화 학습(DeepSeek-R1):** SFT와 RL을 번갈아 사용하는 반복적인 훈련 방식을 채택합니다.
        *   **콜드 스타트(Cold Start):** `Chain-of-Thought (CoT)` 예제를 사용하여 `DeepSeek-V3-Base`를 미세 조정하여 초기 RL 훈련의 불안정성을 완화합니다.
        *   **추론 지향 RL(Reasoning-oriented RL):** 언어 일관성 보상(language consistency reward)을 추가하여 언어 혼합 문제를 해결합니다.
        *   **거부 샘플링(Rejection Sampling) 및 SFT:** 정확한 추론 관련 샘플과 비추론 샘플을 수집하여 모델의 쓰기 및 역할 수행 능력을 향상시킵니다.
        *   **RL 정렬(RL Alignment):** 유용성 및 무해성 측정을 통해 모델을 인간 선호도에 더 잘 맞추고 추론 능력을 개선합니다.

## 결과
DeepSeek-V3 및 DeepSeek-R1 모델은 OpenAI 및 Anthropic과 같은 최첨단 비공개 소스 모델과 비교할 때 필적하는 성능을 달성했으며, 필요한 훈련 리소스는 그 일부에 불과합니다. `MLA`는 `MHA`를 능가하는 성능을 보였고, `GRPO`는 `PPO`와 유사한 성능을 달성하면서도 훨씬 효율적입니다. `DeepSeek-R1-Zero`의 순수 강화 학습은 추론 능력을 자연스럽게 유도하며 지속적인 성능 향상을 보였습니다. `DeepSeek-R1`은 `SFT`와 `RL`을 번갈아 사용하는 반복 훈련을 통해 모델의 가독성 및 언어 혼합 문제를 개선하고 전반적인 성능을 더욱 향상시켰습니다.