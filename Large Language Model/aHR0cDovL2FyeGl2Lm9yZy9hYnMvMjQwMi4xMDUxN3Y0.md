# Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs

Yeonhong Park, Jake Hyun, SangLyul Cho, Bonggeun Sim, Jae W. Lee (2024)

## 🧩 Problem to Solve

최근 Large Language Models (LLMs)의 비약적인 발전에도 불구하고, 모델의 거대한 크기로 인해 발생하는 배포 비용은 여전히 큰 장애물이다. 특히 실제 서비스 환경에서는 쿼리의 특성이나 지연 시간(latency) 제약 조건에 따라 서로 다른 크기의 여러 LLM을 동적으로 선택하여 사용하는 시나리오가 빈번하며, 이는 Speculative Decoding과 같은 가속 기법을 적용할 때도 필수적이다.

그러나 다양한 크기의 LLM들을 동시에 배포하는 것에는 두 가지 주요한 문제가 존재한다. 첫째, 메모리 오버헤드이다. 여러 모델을 개별적으로 배포할 경우 각 모델이 독립적인 메모리 공간을 점유하여 전체 메모리 요구량이 급격히 증가한다. 둘째, 학습 비용이다. 원하는 크기의 모델이 오픈소스로 제공되지 않을 경우, 각 크기별로 모델을 직접 학습시키거나 지식 증류(Knowledge Distillation)를 수행해야 하는데, 이는 막대한 계산 자원과 엔지니어링 노력이 필요하다.

본 논문의 목표는 단일 모델의 메모리 풋프린트만으로 다양한 비트 너비(bit-width)를 가진 여러 LLM을 효율적으로 배포할 수 있는 **Any-Precision LLM** 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 가장 정밀도가 높은 $n$-bit 모델(Parent Model) 하나만 저장하고, 여기서 상위 비트(Most Significant Bits, MSBs)만을 추출함으로써 더 낮은 정밀도의 모델들($(n-1)$-bit, $(n-2)$-bit 등)을 즉석에서 생성하여 사용하는 것이다. 이를 위해 저자는 다음과 같은 핵심 기여를 제시한다.

1.  **Incremental Upscaling (IU)**: 기존의 Any-precision 기법이 요구하던 고비용의 Quantization-Aware Training (QAT) 대신, Post-Training Quantization (PTQ) 프레임워크를 기반으로 한 경량화된 양자화 방법을 제안한다.
2.  **Bitplane-based Software Engine**: 단순히 비트를 절삭하는 것에 그치지 않고, 실제 추론 속도 향상을 위해 메모리 대역폭을 효율적으로 사용하는 Bitplane 기반의 가중치 표현 방식과 전용 GPU 커널을 개발한다.
3.  **Kernel Optimizations**: Bitplane 표현 방식에서 발생하는 오버헤드를 줄이기 위해 가중치 레이아웃 최적화(WLO), 효율적인 Bit-transpose 알고리즘(IBT), 그리고 테이블 룩업 병합(TLM) 기술을 도입한다.

## 📎 Related Works

기존의 LLM 양자화 연구는 주로 weight-only PTQ에 집중되어 왔다. GPTQ는 레이어별 가중치 재구성을 통해 오차를 수정하며, AWQ는 활성화 값에 기반한 스케일링을 통해 중요한 가중치를 보호한다. 또한 SqueezeLLM과 같은 비균등 양자화(Non-uniform Quantization) 방식은 가중치 분포를 더 잘 포착하여 낮은 비트에서도 높은 성능을 보인다.

하지만 기존의 Any-precision DNN 연구는 주로 CNN을 대상으로 하며 모델을 처음부터 다시 학습시켜야 하는 QAT 방식에 의존하므로 LLM에 적용하기 어렵다. 또한, 기존의 양자화 커널들은 대부분 Bitpacking 방식을 사용하는데, 이는 낮은 비트 모델을 실행할 때도 전체 비트 벡터를 로드해야 하므로 메모리 대역폭 절감 효과가 없다는 한계가 있다.

## 🛠️ Methodology

### 1. Incremental Upscaling (IU)
Any-precision 특성을 유지하면서 모델을 생성하기 위해 본 논문은 **Incremental Upscaling** 과정을 제안한다.

*   **Step 1 (Seed Model Generation)**: 지원하고자 하는 최소 비트 너비($n_1$)로 모델을 양자화하여 Seed Model을 생성한다.
*   **Step 2 (Incremental Upscaling)**: Seed Model에서 시작하여 한 번에 1비트씩 정밀도를 높여 최종 $n_K$-bit Parent Model에 도달한다. 이때 $n_i$-bit 모델의 파라미터를 그대로 상속받고, 끝에 1비트를 추가하는 방식으로 진행한다.

이 과정에서 저자는 SqueezeLLM의 클러스터링 기반 비균등 양자화를 백본으로 사용한다. 구체적으로, $n_i$ 비트의 각 클러스터를 가중치 K-means 클러스터링을 통해 두 개의 서브 클러스터로 분할함으로써 $n_{i+1}$ 비트 모델로 확장한다.

### 2. Specialized Software Engine
추론 시 메모리 대역폭을 실제로 절약하기 위해 **Bitplane-based Representation**을 도입한다. 이는 가중치를 비트 위치별로 분리하여 저장하는 방식으로, $k$-bit 모델이 필요할 경우 상위 $k$개의 비트플레인만 메모리에서 로드하면 된다.

#### 추론 파이프라인 (Thread-Level Operations)
1.  **Load**: 입력 활성화 값과 필요한 비트의 가중치 비트플레인을 로드한다.
2.  **Bit-transpose**: 로드된 비트플레인 데이터를 가중치별로 연속적인 비트가 배치되도록 재배열한다.
3.  **Index Extraction**: 시프트 및 마스킹 연산을 통해 Centroid Table에서 값을 찾기 위한 인덱스를 생성한다.
4.  **Dequantization**: 공유 메모리에 저장된 Centroid Table에서 실제 FP16 값을 가져온다.
5.  **MAC**: 역양자화된 가중치와 입력 활성화 값을 곱하고 누적한다.

### 3. GPU Kernel Optimization
Bitplane 방식의 효율성을 극대화하기 위해 세 가지 최적화를 수행한다.

*   **Weight Bitplane Layout Optimization (WLO)**: GPU의 메모리 접근 효율(Coalesced Access)을 높이기 위해 비트플레인 내의 바이트 순서를 변경하여, 워프(warp) 내 스레드들이 연속적인 활성화 값 블록에 접근하도록 한다.
*   **Efficient Bit-transpose (IBT)**: 32비트 벡터를 SIMD처럼 처리하여 비트 전치 연산을 최적화한다. 기존 알고리즘보다 비트 연산 횟수를 획기적으로 줄여(예: 76회 $\rightarrow$ 40회) 오버헤드를 낮춘다.
*   **Merging Table Lookups (TLM)**: 3-bit 모델의 경우, 두 개의 3-bit 인덱스를 하나의 6-bit 인덱스로 병합하여 테이블 룩업 횟수를 절반으로 줄인다.

## 📊 Results

### 1. 메모리 절감 효과
Llama-2-7B 모델을 기준으로 3~8비트의 모든 모델을 동시에 지원할 때, 개별 배포 방식은 29.9 GB가 필요하지만 Any-Precision LLM은 8.4 GB만으로 가능하여 최대 **3.56배의 메모리 절감**을 달성하였다.

### 2. 모델 품질 (Quality)
SqueezeLLM을 기반으로 한 IU 방식은 각 비트 너비에서 독립적으로 양자화된 SOTA 모델들과 거의 동일한 성능을 보였다.
*   **Perplexity**: WikiText2, C4, PTB 데이터셋에서 독립 양자화 모델과의 차이가 대부분 0.1 미만으로 매우 적었다.
*   **Zero-shot Accuracy**: 5가지 벤치마크 태스크 평균 정확도에서 독립 양자화 모델 대비 오차가 0.2% 이내로 유지되었다.

### 3. 추론 성능 (Throughput)
RTX 4090, RTX 4070 Laptop, Jetson AGX Orin 등 다양한 환경에서 평가한 결과, 본 제안 커널은 기존 SqueezeLLM 커널과 대등하거나(RTX 시리즈), Jetson 환경에서는 훨씬 뛰어난 성능을 보였다. 특히 비트 너비가 낮아질수록 FP16 대비 속도 향상 폭이 선형적으로 증가하는 것을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Any-precision 개념을 LLM에 성공적으로 이식하며, 특히 비균등 양자화와 Bitplane 표현 방식의 결합이 필수적임을 입증하였다.

**비균등 양자화의 중요성**: 저자는 GPTQ나 AWQ와 같은 균등 양자화(Uniform Quantization) 방식에 IU를 적용했을 때 심각한 품질 저하가 발생함을 보였다. 이는 균등 양자화의 경우 비트 확장 시 가중치 보정 과정에서 발산(Divergence)이 발생하고, 이를 막기 위한 Clamping 연산이 오히려 오차를 증폭시키기 때문이다. 반면 클러스터링 기반의 비균등 양자화는 단순히 클러스터를 분할하는 방식이기에 이러한 불안정성 없이 정밀도를 높일 수 있다.

**한계 및 논의**: 본 커널은 Tensor Core를 사용하지 않으므로, $M$ 값이 매우 큰 Prefill 단계(Matrix-Matrix Multiplication)에서는 cuBLAS FP16 커널보다 느려지는 경향이 있다. 이를 해결하기 위해 $M$이 특정 임계값(예: 16)을 넘으면 별도의 역양자화 커널 후 cuBLAS를 사용하는 하이브리드 방식을 채택하였다.

## 📌 TL;DR

본 연구는 단일 $n$-bit 부모 모델로부터 하위 정밀도 모델들을 즉석에서 추출하여 사용할 수 있는 **Any-Precision LLM** 프레임워크를 제안한다. **Incremental Upscaling**을 통해 학습 비용 없이 SOTA 수준의 품질을 가진 다중 정밀도 모델을 생성하고, **Bitplane 기반 전용 GPU 엔진**을 통해 메모리 풋프린트를 획기적으로 줄이면서도 추론 속도를 최적화하였다. 이 기술은 메모리 자원이 제한된 온디바이스(On-device) 환경에서 다양한 지연 시간 요구사항을 충족해야 하는 LLM 서비스 배포에 매우 유용한 해결책이 될 것이다.