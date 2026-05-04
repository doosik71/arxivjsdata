# MoR: Mixture of Ranks for Low-Rank Adaptation Tuning

Chuanyu Tang, Yilong Chen, Zhenyu Zhang, Junyuan Shang, Wenyuan Zhang, Yong Huang, Tingwen Liu (2024)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)의 효율적인 미세 조정(Fine-tuning)을 위해 널리 사용되는 Low-Rank Adaptation(LoRA)의 한계를 해결하고자 한다. 구체적으로 해결하려는 문제는 다음과 같다.

첫째, LoRA의 Rank 크기를 단순히 늘리는 것만으로는 고차원 정보(high-rank information)를 효과적으로 캡처하지 못하며, 이는 성능의 병목 현상으로 이어진다. 

둘째, 최근 제안된 MoE(Mixture of Experts) 스타일의 LoRA 방법론들은 성능을 높일 수는 있으나, 파라미터 수가 크게 증가하고 추론 지연 시간(inference latency)이 늘어나는 문제가 있다. 이는 효율적인 미세 조정과 쉬운 적용이라는 LoRA의 본래 목적에 배치된다.

따라서 본 연구의 목표는 높은 파라미터 효율성을 유지하면서도 모델의 다중 작업(multi-task) 수행 능력을 강화하는 새로운 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"공유된 파라미터 서브스페이스를 학습하고, 이를 수학적 변환을 통해 여러 하위 작업 공간으로 매핑하는 것"**이다.

저자들은 여러 개의 LoRA를 통합하는 것이 결과적으로 LoRA의 Rank를 확장하는 것과 동일하다는 해석 프레임워크를 제시한다. 하지만 Rank를 직접 높이는 대신, 낮은 Rank의 LoRA가 이미 충분한 내재적 정보(intrinsic information)를 가지고 있다는 가설을 세운다. 이를 바탕으로, 적은 수의 공유 파라미터에 다차원 스케일링 변환(multi-dimensional scaling transformations)을 적용하여 고차원 정보를 유도함으로써, 학습 난이도를 낮추고 다중 작업 능력을 향상시킨 Mixture of Ranks(MoR)를 제안한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 MoR과의 차이점을 설명한다.

1.  **LoRA 및 변형 모델들**: LoRA, VeRA, Tied-LoRA, DoRA 등이 언급된다. 특히 DoRA는 가중치를 방향(direction)과 크기(amplitude)로 분해하여 성능을 높였으나, 여전히 단일 방향의 변환만을 수행하므로 다중 작업 학습 능력이 부족하다는 한계가 있다.
2.  **MoE-style PEFT**: MoELoRA, LoRAMoE 등이 제안되었다. 이들은 여러 개의 LoRA 전문가(experts)를 두고 라우터를 통해 선택적으로 활성화함으로써 모델 용량을 늘린다. 그러나 전문가마다 독립적인 $A, B$ 행렬을 가지므로 파라미터 중복(redundancy)이 심하고 효율성이 떨어진다.

MoR은 공유된 $A, B$ 행렬을 사용하면서 스케일링 벡터만을 다르게 가져감으로써, MoE의 표현력과 LoRA의 효율성을 동시에 달성한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
MoR은 크게 **공유 전문가(Shared Experts)**, **다중 랭크 적응(Multi-rank Adaptation)**, **믹스처 학습(Mixture Learning)**의 세 가지 구성 요소로 이루어져 있다. 모델의 Feed-Forward Network(FFN) 모듈을 MoR 플러그인으로 대체하며, 기본 모델의 가중치는 동결(freeze)한 상태로 MoR 파라미터만 학습한다.

### 주요 구성 요소 및 작동 원리

1.  **LoRA 변환 (Transform of LoRA)**:
    공유된 LoRA 행렬 $A$와 $B$에 대해, 학습 가능한 대각 행렬(diagonal matrix) $\Lambda_A$와 $\Lambda_B$를 도입하여 스케일링 변환을 수행한다.
    $$\hat{A} = \Lambda_A A, \quad \hat{B} = \Lambda_B B$$
    여기서 $\Lambda_A = \text{diag}(\lambda^A_0, \dots, \lambda^A_{r-1})$ 이고 $\Lambda_B = \text{diag}(\lambda^B_0, \dots, \lambda^B_{d_{out}-1})$ 이다. 이 변환을 통해 파라미터 공간 내의 특정 서브스페이스에서 LoRA의 랭크를 변형하여 고차원 정보를 학습할 수 있게 한다.

2.  **MoR 아키텍처 및 라우팅**:
    각 전문가 $i$에 대해 다음과 같은 변환 식 $D_i$를 정의한다.
    $$D_i = \frac{\alpha}{r} \Lambda_{B,i} B \Lambda_{A,i} A$$
    이후 입력 $x$에 대해 라우터 $G_i(x)$가 각 전문가의 가중치를 계산하며, 최종 출력은 다음과 같이 결정된다.
    $$o = Wx + \sum_{i=1}^{N} G_i(x) D_i x$$
    여기서 $G_i(x)$는 Softmax 함수를 통해 계산된 가중치이다.

3.  **효율적 구현 (Implementation in Llama)**:
    학습 및 추론 속도를 높이기 위해, 개별 벡터 $\lambda$들을 행렬 $\Omega_A \in \mathbb{R}^{N \times r}$와 $\Omega_B \in \mathbb{R}^{N \times d_{out}}$ 형태로 스택(stack)하여 GPU의 병렬 연산을 활용한다. 최종 연산식은 다음과 같다.
    $$o = Wx + \frac{\alpha}{r} \sum_{i=1}^{N} G_i(x) \Omega_B \cdot B (\Omega_A \cdot (Ax))$$

## 📊 Results

### 실험 설정
- **데이터셋**: Tulu-v2의 서브셋을 사용하여 다중 작업 학습을 수행하였다.
- **평가 지표**: Commonsense & Reading Comprehension, Language Modeling, World Knowledge 등 3가지 카테고리의 11개 벤치마크에서 0-shot 정확도를 측정하였다.
- **비교 대상**: Full SFT, Vanilla LoRA, DoRA, MoELoRA.

### 주요 결과
- **성능 향상**: MoR은 LoRA 대비 평균 7.4%, DoRA 대비 7.21% 성능 향상을 보였으며, MoELoRA보다도 1.31% 더 높은 성능을 기록하였다. 특히 Full SFT 성능의 98% 수준에 도달하면서도 사용하는 파라미터는 전체의 0.34%에 불과하였다.
- **파라미터 효율성**: MoELoRA와 비교했을 때, MoR은 더 적은 파라미터를 사용하면서도 더 높은 성능을 낸다. 전문가 수($N$)가 증가함에 따라 MoELoRA는 파라미터 수가 급격히 증가하지만, MoR은 매우 완만하게 증가하여 확장성(scalability)이 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

### 분석 및 논의
1.  **전문가 수와 Rank의 영향**: 실험 결과, 전문가 수는 8개일 때 최적의 성능을 보였으며, 12개를 초과하면 오히려 성능이 정체되거나 하락하는 경향을 보였다. 공유 LoRA의 Rank는 32일 때 가장 효과적이었으며, 64를 초과하면 기존 LoRA와 마찬가지로 성능 저하가 발생하였다.
2.  **라우터 전략**: 단순 평균(Mean Pooling)이나 밸런스 손실(Balanced Router)을 사용하는 것보다, 학습 가능한 Softmax 라우터를 사용하는 것이 가장 높은 성능을 보였다. 이는 모델이 입력에 따라 최적의 전문가를 유연하게 선택하는 것이 중요함을 시사한다.
3.  **작업별 전문가 특성**: 라우터 가중치 시각화 결과, MMLU나 SiQA 등 서로 다른 작업마다 활성화되는 전문가가 서로 다르게 나타났다. 이는 MoR의 각 전문가가 특정 작업에 특화된 지식을 캡처하고 있음을 의미한다.

### 한계점
- **모델 크기의 제한**: 자원 제약으로 인해 7B 모델에서만 실험이 진행되었으므로, 더 큰 모델에서의 효과는 추가 검증이 필요하다.
- **추론 지연**: Vanilla LoRA와 달리 MoE 구조를 가지므로, 학습된 파라미터를 기본 모델 가중치에 병합(merge)할 수 없다. 이로 인해 추론 시 추가적인 지연 시간이 발생하며, 향후 룰 기반 라우터 등을 통해 이를 해결할 필요가 있다.

## 📌 TL;DR

본 논문은 LoRA의 Rank 확장 시 발생하는 성능 병목과 MoE-LoRA의 파라미터 과다 문제를 동시에 해결하는 **Mixture of Ranks (MoR)** 프레임워크를 제안한다. MoR은 **공유된 LoRA 행렬에 가벼운 스케일링 변환 벡터들을 적용**하여 효율적으로 고차원 정보를 학습하며, 이를 통해 **최소한의 파라미터 증가만으로 Full SFT에 근접하는 다중 작업 성능**을 달성하였다. 이 연구는 효율적인 LLM 어댑터 설계에 있어 파라미터 공유와 동적 변환의 결합이 매우 효과적임을 보여준다.