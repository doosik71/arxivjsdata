# OmniVLM: A Token-Compressed, Sub-Billion-Parameter Vision-Language Model for Efficient On-Device Inference

Wei Chen, Zhiyuan Li, Shuo Xin (2024)

## 🧩 Problem to Solve

본 논문은 엣지 디바이스(스마트폰, 노트북, 임베디드 시스템 등)에서 Vision-Language Model(VLM)을 효율적으로 배포하기 위해 발생하는 세 가지 주요 도전 과제를 해결하고자 한다.

첫째, **시각적 입력 토큰화로 인한 과도한 연산 오버헤드**이다. 고해상도 이미지를 처리할 때 생성되는 대량의 시각적 토큰은 연산량과 메모리 사용량을 급격히 증가시킨다. 둘째, **에너지 제한 환경에서의 전력 소비 문제**이다. 특히 파라미터 수가 많은 모델은 토큰당 전력 소모가 커서 모바일 기기의 배터리 수명에 치명적인 영향을 미친다. 셋째, **소형 모델의 성능 저하**이다. 2B 파라미터 미만의 기존 VLM들은 복잡한 시각적 이해 및 추론 능력(예: MMMU 벤치마크)이 대형 모델에 비해 현저히 낮다는 한계가 있다.

결과적으로 본 연구의 목표는 연산 효율성, 에너지 소비, 그리고 모델 성능 사이의 최적의 트레이드-오프를 달성하여, 1B 파라미터 미만의 크기로도 높은 성능을 내며 실시간 온디바이스 추론이 가능한 VLM을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **시각적 토큰 압축 메커니즘(Token Compression Mechanism)**: 이미지 토큰의 수를 729개에서 81개로 약 9배 감소시키면서도 시각적-의미적 충실도(visual-semantic fidelity)를 유지하는 효율적인 투영 계층(Projection Layer)을 설계하였다.
2. **Minimal-Edit DPO 프레임워크**: Direct Preference Optimization(DPO)를 적용하되, 교사 모델이 기본 모델의 출력물에 최소한의 수정(minimal edits)만을 가한 쌍을 학습하게 함으로써, 모델의 안정성을 해치지 않으면서 응답의 정확도와 품질을 향상시켰다.
3. **효율적인 온디바이스 추론 구현**: 968M라는 소형 파라미터 규모로 기존의 소형 VLM(nanoLLAVA 등)보다 우수한 벤치마크 성능을 기록함과 동시에, 실제 엣지 디바이스에서 비약적으로 빠른 응답 속도(TTFT)와 디코딩 속도를 달성하였다.

## 📎 Related Works

### VLM의 진화 및 아키텍처

LLaVA와 같은 모델들이 시각 인코더와 언어 모델을 결합하고 GPT-4를 이용한 멀티모달 명령어 튜닝 데이터를 생성함으로써 일반적인 시각-언어 이해 능력을 비약적으로 발전시켰다. 이후 PaliGemma, Qwen2-VL, Llama 3.2-Vision, InternVL 등 다양한 산업계 및 학계의 모델들이 등장하며 성능을 높여왔으나, 대부분은 엣지 디바이스에 올리기에는 여전히 무거운 규모를 가지고 있다.

### 엣지 배포 및 추론 최적화

제한된 메모리와 연산 능력 환경에서 LLM을 구동하기 위해 양자화(Quantization), 가지치기(Pruning), KV 캐시 최적화 등이 사용되고 있다. GGML 라이브러리나 llama.cpp, Ollama, MLC와 같은 소프트웨어 스택이 로컬 실행을 가능하게 하고 있으나, VLM의 경우 텍스트뿐만 아니라 거대한 시각적 토큰 셋을 처리해야 하므로 추가적인 최적화가 필요하다.

## 🛠️ Methodology

### 전체 시스템 아키텍처

OmniVLM은 기본적으로 LLaVA 아키텍처를 확장하여 설계되었으며, 다음과 같은 구성 요소로 이루어져 있다.

- **Base Language Model**: $\text{Qwen2.5-0.5B-Instruct}$ 모델을 사용하여 텍스트 입력에 대한 강한 문맥 이해 능력을 확보하였다.
- **Vision Encoder**: $\text{SigLIP-400M}$을 사용하여 $384 \times 384$ 해상도의 이미지로부터 고품질 임베딩을 생성한다. 패치 크기는 $14 \times 14$이다.
- **Projection Layer (Projector)**: MLP 기반의 투영 계층으로, 시각 인코더의 출력 임베딩을 언어 모델의 토큰 공간으로 정렬한다. 이 계층에서 핵심적인 토큰 압축이 수행된다.

### 이미지 토큰 압축 (Image Token Compression)

SigLIP-384 모듈은 기본적으로 $27 \times 27$ 공간 그리드에서 729개의 토큰을 생성한다. OmniVLM은 이를 $[batch\_size, 729, hidden\_size]$에서 $[batch\_size, 81, hidden\_size]$로 변환하여 토큰 수를 9배 줄인다.

저자들은 세 가지 변환 기법을 비교 분석하였다.

1. **Reshaping**: 단순 차원 재구성.
2. **1D Convolution**: 커널 크기 $k=9$, 스트라이드 $s=9$ 적용.
3. **2D Convolution**: 커널 크기 $(9, 1)$, 스트라이드 $(9, 1)$ 적용.

실험 결과, Convolution 기반 방식보다 **Reshaping 전략**이 더 낮은 검증 손실(Validation Loss)과 Perplexity를 기록하였다. 또한, 압축률을 729, 243, 81, 9개 토큰으로 변경하며 테스트한 결과, **81개 토큰** 설정이 연산 효율성과 모델 성능 사이의 최적의 균형점을 제공함을 확인하였다.

### Minimal-Edit DPO

응답 품질을 높이기 위해 Direct Preference Optimization(DPO)를 적용하였다. 일반적인 DPO와 달리 'Minimal-Edit' 방식을 사용하여, 교사 모델이 기존 모델의 출력에서 정답에 결정적인 영향을 주는 최소한의 부분만 수정하여 '선택됨(chosen)'과 '거부됨(rejected)' 쌍을 생성한다. 이를 통해 모델의 기존 행동 양식을 급격하게 바꾸지 않으면서도 정밀한 품질 향상을 꾀하였다.

### 다단계 학습 절차 (Multi-Stage Training)

1. **Pretraining (사전 학습)**: 대규모 이미지-캡션 데이터셋(약 558K 샘플)을 사용한다. 시각적-언어적 정렬을 위해 **Projector만 학습**시키며, 시각 인코더와 LLM은 동결한다.
2. **Supervised Fine-Tuning (SFT)**: LLaVA, UniMM-Chat 및 내부 데이터셋(총 6M 샘플)을 사용해 질의응답 쌍을 학습한다. 이 단계에서는 **Projector와 LLM 부분을 학습**시키고 시각 인코더는 동결한다.
3. **Direct Preference Optimization (DPO)**: RLAIF-V 프로젝트 데이터를 활용하여 선호도 학습을 수행한다. **Projector와 LLM backbone을 학습**시키며 시각 인코더는 계속 동결 상태를 유지한다.

## 📊 Results

### 품질 벤치마크 성능

OmniVLM(968M)은 기존의 소형 VLM인 nanoLLAVA와 비교하여 모든 지표에서 우수한 성능을 보였다.

| Benchmark | OmniVLM | nanoLLAVA |
| :--- | :---: | :---: |
| ScienceQA (Eval/Test) | **71.0** | 59.0 |
| POPE | **93.3** | 84.1 |
| MM-VET | **30.9** | 23.9 |
| MMMU (Test) | **42.1** | 28.6 |
| MMMU (Eval) | **40.0** | 30.4 |

특히 MMMU Test에서 13.5 포인트의 큰 격차를 보이며 소형 모델임에도 강력한 멀티모달 추론 능력을 입증하였다.

### 엣지 디바이스 추론 성능

실제 하드웨어에서 Time-to-First-Token(TTFT)과 Decoding Speed를 측정한 결과는 다음과 같다.

**1. Laptop (AMD Ryzen AI)**

- **TTFT**: OmniVLM(0.75s) $\gg$ nanoLLAVA(6.82s) $\rightarrow$ 약 9배 빠름.
- **Decoding Speed**: OmniVLM(29.41 tok/s) $>$ nanoLLAVA(19.20 tok/s) $\rightarrow$ 약 1.5배 빠름.

**2. Mobile (Google Pixel 6 / Samsung S22)**

- **TTFT**: OmniVLM(7.48s) $\gg$ nanoLLAVA(60.23s) $\rightarrow$ 약 8배 빠름.
- **Decoding Speed**: OmniVLM(31.88 tok/s) $>$ nanoLLAVA(24.33 tok/s) $\rightarrow$ 약 1.3배 빠름.

## 🧠 Insights & Discussion

본 논문의 가장 흥미로운 통찰은 **적절한 수준의 토큰 압축이 오히려 소형 모델의 성능을 향상시킬 수 있다**는 점이다. 실험 결과, 토큰 수를 그대로 유지(729개)했을 때보다 81개로 압축했을 때 검증 손실이 더 낮게 나타났다. 이는 파라미터 규모가 작은 언어 모델의 경우, 너무 긴 시각적 토큰 시퀀스가 입력되면 Attention 메커니즘이 포화(Saturation)되거나 문맥 윈도우 제약으로 인해 성능이 저하될 수 있음을 시사한다. 즉, 소형 VLM에서는 공격적인 토큰 압축이 연산량 감소뿐만 아니라 모델의 인지 효율성을 높이는 전략이 될 수 있다.

다만, 본 논문은 주로 CPU/GPU 추론 성능에 집중하고 있으며, 향후 NPU(Neural Processing Unit) 최적화에 대한 구체적인 방법론은 미래 작업(Future Work)으로 남겨두었다. 또한, 사용된 데이터셋의 세부 구성 중 일부(Nexa AI 내부 데이터셋)는 공개되지 않아 완전한 재현에는 한계가 있을 수 있다.

## 📌 TL;DR

OmniVLM은 968M 파라미터 규모의 초경량 VLM으로, **시각적 토큰을 9배 압축(729 $\to$ 81)**하고 **Minimal-Edit DPO**를 통해 응답 품질을 최적화하였다. 그 결과, nanoLLAVA보다 뛰어난 벤치마크 성능을 보이면서도 온디바이스 환경(노트북, 모바일)에서 **최대 9배 빠른 응답 속도**를 달성하였다. 이 연구는 소형 VLM에서 토큰 압축이 연산 효율과 모델 성능을 동시에 잡을 수 있는 핵심 기법임을 보여주며, 향후 NPU 최적화를 통해 실제 모바일 AI 에이전트 구현에 중요한 기반이 될 것으로 보인다.
