# PTQTP: Post-Training Quantization to Trit-Planes for Large Language Models

He Xiao, Runming Yang, Qingyao Yang, Wendong Xu, Zhen Li, Yupeng Su, Zhengwu Liu, Hongxia Yang, Ngai Wong (2026)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)을 매우 낮은 비트 너비(extremely low bit-widths)로 양자화할 때 발생하는 연산 효율성과 표현 능력(representational capacity) 사이의 근본적인 트레이드오프 문제를 해결하고자 한다.

기존의 초저비트 양자화 방식은 크게 두 가지 한계를 가진다. 첫째, Binary PTQ(1-bit)와 같은 방식은 가중치를 $\pm 1$로 강제함으로써 거의 0에 가까운 가중치(노이즈)까지 증폭시키는 '강제 활성화(forced activation)' 문제를 일으키며, 이는 수학적 추론이나 코딩과 같은 정밀한 논리 흐름이 필요한 작업에서 성능 붕괴를 초래한다. 둘째, BitNet과 같은 1.58-bit Ternary 방식은 높은 성능을 보이지만, 양자화 인식 학습(Quantization-Aware Training, QAT)에 의존하여 막대한 재학습 리소스와 시간이 소요된다.

따라서 본 연구의 목표는 재학습이나 미세 조정(fine-tuning) 없이도 QAT 수준의 성능을 내면서, 연산 효율성을 극대화한 구조적 Post-Training Quantization(PTQ) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 가중치 행렬을 **Magnitude(크기)**와 **Topology(구조)**로 분리하여 설계하는 **Magnitude-Topology Decoupling** 전략이다.

핵심 설계는 가중치 행렬 $W$를 두 개의 Ternary Trit-Planes($\{-1, 0, 1\}$ 값만 가지는 평면)의 선형 결합으로 분해하는 것이다. 여기서 "0" 상태의 도입은 불필요한 노이즈 특징을 선택적으로 제거하는 암시적 디노이징(implicit denoising) 역할을 수행한다. 또한, 두 개의 Trit-Planes를 사용하는 'Dual' 구조를 통해 첫 번째 평면이 거친 구조(Coarse Structure)를 잡고, 두 번째 평면이 세부적인 잔차(Fine-grained Correction)를 보정함으로써 표현력을 높였다. 결과적으로 무거운 곱셈 연산을 가벼운 덧셈 연산으로 대체하여 추론 속도를 획기적으로 향상시켰다.

## 📎 Related Works

논문에서는 기존의 PTQ 및 Ternary 양자화 연구를 다음과 같이 구분하여 설명한다.

1. **범용 PTQ (4-bit 이상):** GPTQ, AWQ 등은 높은 정확도를 달성했지만, 여전히 하드웨어 수준에서 비용이 많이 드는 Multiply-Accumulate(MAC) 연산에 의존한다.
2. **초저비트 PTQ (sub-4bit):** PBLLM, SliM-LLM, AQLM 등은 아웃라이어 처리나 벡터 양자화를 통해 성능을 높이려 했으나, 비정형적인 가중치 분류 방식을 사용하여 하드웨어 구현이 복잡하거나 추론 시 디코딩 오버헤드가 발생한다.
3. **Ternary QAT:** BitNet(1.58-bit)은 매우 효율적이고 강력하지만, 사전 학습 단계부터 양자화를 적용해야 하므로 일반적인 pretrained 모델에 적용하기 어렵고 학습 비용이 매우 크다.

PTQTP는 이러한 기존 방식들과 달리, **구조적인 Ternary 분해**를 통해 QAT의 성능과 PTQ의 편의성을 동시에 확보하며, 하드웨어 친화적인 덧셈 기반 추론을 가능하게 한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

PTQTP는 가중치 행렬 $W \in \mathbb{R}^{n \times d}$를 다음과 같은 수식으로 근사한다.

$$W \approx \hat{W} = \sum_{k=1}^{2} \text{diag}(\alpha^{(k)}) \cdot T^{(k)}$$

여기서 $T^{(k)} \in \{-1, 0, 1\}^{n \times d}$는 이산적인 라우팅 구조를 나타내는 **Trit-Plane**이며, $\alpha^{(k)} \in \mathbb{R}^n$은 각 채널의 이득을 조절하는 연속적인 **Scaling coefficients(Magnitude)**이다.

### 2. 최적화 절차: Dual Trit-Planes Approximation

가중치를 근사하기 위해 $\alpha$와 $T$를 교대로 최적화하는 과정을 거친다.

- **Scaling coefficients ($\alpha$) 최적화:** $T$가 고정된 상태에서, 적응형 릿지 회귀(Adaptive Ridge Regression)를 사용하여 최적의 $\alpha$를 찾는다.
  - 로컬 베이시스 행렬 $S_i = [(T^{(1)}_i)^T, (T^{(2)}_i)^T] \in \mathbb{R}^{d \times 2}$를 구성한다.
  - 다음의 폐쇄형 해(closed-form solution)를 통해 $\alpha_i$를 계산한다.
    $$\alpha_i = (S_i^T S_i + \lambda_i I)^{-1} S_i^T W_i^T$$
  - 여기서 $\lambda_i$는 수치적 안정성을 위한 정규화 파라미터이다.

- **Trit-Planes ($T$) 최적화:** $\alpha$가 고정된 상태에서, 각 원소 $W_{ij}$에 대해 다음의 제곱 오차를 최소화하는 $\{-1, 0, 1\}$ 조합을 지역적으로 전수 조사(local exhaustive search)하여 업데이트한다.
    $$T^{(k)}_{ij} = \arg \min_{c^{(k)}_m \in \{-1, 0, 1\}} \left( W_{ij} - \sum_{k=1}^{2} \alpha^{(k)}_i c^{(k)}_m \right)^2$$

### 3. Progressive Optimization 및 Adaptive Regularization

수치적 불안정성을 해결하기 위해 **조건수(Condition Number)** $\kappa_{i, \text{approx}}$를 기반으로 $\lambda_i$를 동적으로 업데이트한다. $\kappa$ 값이 특정 임계값($10^6$)을 넘으면 $\lambda$를 증가시켜 해의 발산을 막는다. 이 과정을 최대 $T_{\max}=50$회 반복하며, Frobenius norm $\|W - \hat{W}\|_F$가 단조 감소하도록 보장하여 지역 최솟값으로 수렴시킨다.

### 4. 추론 가속화 (Inference Efficiency)

가중치가 $\{-1, 0, 1\}$로 제한되므로, 실제 추론 시 곱셈 연산을 수행하지 않는다.

- **LUT-GEMM:** 가중치 패턴과 활성화 벡터 간의 모든 가능한 내적 값을 미리 계산하여 룩업 테이블(LUT)에 저장한다.
- **Additive Inference:** 실제 연산은 LUT 참조와 덧셈으로만 이루어지며, 이는 부동 소수점 곱셈기를 제거하고 하드웨어의 덧셈기만을 사용하여 연산 강도를 $O(1)$로 낮춘다.

## 📊 Results

### 1. 실험 설정

- **대상 모델:** LLaMA 2, LLaMA 3.x, Qwen 3 (0.6B ~ 70B)
- **비교 대상:** GPTQ, AWQ, AQLM, PBLLM, SliM-LLM 및 1.58-bit QAT (BitNet)
- **측정 지표:** Perplexity (WikiText-2, C4), Zero-shot Reasoning (MMLU, ARC 등), 수학 및 코딩 벤치마크 (Math-500, GSM8K, HumanEval)

### 2. 주요 결과

- **언어 모델링 성능:** Table 1에서 확인되듯, PTQTP는 거의 모든 모델 크기에서 기존의 1-3 bit PTQ 방법들보다 낮은 Perplexity를 기록하며, 일부 4-bit 방법과 대등한 성능을 보인다.
- **추론 능력 유지:** Table 2와 3에서 PTQTP는 FP16 대비 약 95%의 성능 유지율을 보였다. 특히 기존의 2-bit 미만 PTQ 방식들이 수학(Math-500) 및 코딩 작업에서 성능이 완전히 붕괴(0%에 가까운 정확도)되는 것과 달리, PTQTP는 매우 낮은 성능 저하만을 보이며 강력한 추론 능력을 유지했다.
- **QAT 대비 효율성:** 1.58-bit QAT(BitNet)와 대등하거나 오히려 능가하는 성능을 보이면서도, 양자화에 소요되는 시간은 10-14 GPU-days에서 **단 1시간 내외**로 단축시켰다.
- **추론 속도:** NVIDIA RTX 3090 GPU 기준, LLaMA-2 7B 모델에서 FP16 대비 **4.63배의 end-to-end 디코드 속도 향상**을 달성했다.

## 🧠 Insights & Discussion

### 강점

PTQTP는 초저비트 양자화에서 가장 치명적인 문제였던 '추론 능력의 붕괴'를 Magnitude-Topology 분리라는 단순하면서도 강력한 구조적 접근법으로 해결했다. 특히 재학습 없이(PTQ) 1.58-bit 수준의 성능을 낸다는 점과, LUT-GEMM을 통해 실질적인 하드웨어 가속을 증명한 점이 매우 고무적이다.

### 한계 및 논의사항

- **메모리 레이아웃:** 현재는 Trit-Plane을 저장하기 위해 2-bit 데이터 타입을 사용하는데, 이를 더 세밀하게 비트 패킹(bit-packing)한다면 캐시 효율성을 20-30% 더 높일 수 있을 것으로 보인다.
- **하드웨어 의존성:** 현재는 CUDA 커널(LUT-GEMM)을 통해 가속화했지만, 본 논문이 제안하는 덧셈 기반 연산의 잠재력을 완전히 끌어내기 위해서는 ASIC나 FPGA 같은 전용 하드웨어 가속기와의 소프트웨어-하드웨어 공동 설계(Co-design)가 필수적이다.
- **Trit-Plane의 개수:** 본 논문은 Dual(2개) 평면을 사용했으나, 평면의 개수를 늘렸을 때의 성능 향상 폭과 연산 비용 사이의 상관관계에 대한 추가 분석이 필요하다.

## 📌 TL;DR

PTQTP는 LLM의 가중치를 두 개의 Ternary Trit-Planes($\{-1, 0, 1\}$)와 스케일링 계수로 분해하는 새로운 구조적 PTQ 프레임워크이다. 재학습 없이 1.58-bit 양자화를 구현하여 **QAT 수준의 성능을 유지하면서도 양자화 시간을 획기적으로 줄였으며, 덧셈 기반의 추론을 통해 FP16 대비 최대 4.63배의 속도 향상**을 이루었다. 특히 기존 초저비트 양자화에서 무너졌던 수학적 추론 및 코딩 능력을 보존함으로써, 리소스 제한 환경에서의 실용적인 LLM 배포를 위한 새로운 기준을 제시했다.
