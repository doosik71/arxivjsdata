# Mamba-X: An End-to-End Vision Mamba Accelerator for Edge Computing Devices

Dongho Yoon, Gungyu Lee, Jaewon Chang, Yunjae Lee, Dongjae Lee, Minsoo Rhu (2025)

## 🧩 Problem to Solve

본 논문은 엣지 컴퓨팅 장치에서 Vision Mamba 모델을 효율적으로 배포하기 위한 하드웨어 가속기 설계 문제를 다룬다.

최근 Transformer 기반의 비전 모델(ViT 등)은 입력 시퀀스 길이 $L$에 대해 계산 및 메모리 요구량이 이차 복잡도 $O(L^2)$로 증가하는 한계가 있다. 이를 해결하기 위해 선형 복잡도 $O(L)$를 갖는 State Space Models (SSMs) 기반의 Vision Mamba가 제안되었으며, 이는 특히 고해상도 이미지 처리에서 낮은 지연 시간과 메모리 소비라는 이점을 제공한다.

그러나 Vision Mamba를 자원이 제한된 엣지 장치(예: NVIDIA Jetson AGX Xavier)에 배포할 때 심각한 성능 저하가 발생한다. 그 핵심 이유는 Vision Mamba의 핵심 연산인 **Selective Scan**이 본질적으로 순차적인(sequential) 데이터 의존성을 갖기 때문이다. 이러한 특성은 고도의 병렬 처리에 최적화된 GPU 아키텍처에서 계산 자원의 저활용(under-utilization)을 초래하며, 제한된 온칩(on-chip) SRAM 용량으로 인해 중간 상태 변수들이 오프칩(off-chip) 메모리로 빈번하게 유출(spill)되어 과도한 메모리 트래픽을 유발한다. 따라서 본 논문의 목표는 이러한 계산 병목과 메모리 트래픽 문제를 해결하여 엣지 환경에서 Vision Mamba를 효율적으로 가속하는 end-to-end 가속기 Mamba-X를 설계하는 것이다.

## ✨ Key Contributions

Mamba-X의 중심 설계 아이디어는 **하드웨어-알고리즘 공동 설계(Hardware-Algorithm Co-design)**를 통해 Selective Scan의 순차적 의존성을 극복하고 메모리 효율을 극대화하는 것이다.

1. **Systolic Scan Array (SSA) 설계**: 병렬 prefix sum 알고리즘인 Kogge-Stone 알고리즘을 시스톨릭 어레이(Systolic Array) 구조로 재구성하였다. 이를 통해 인접한 처리 요소(PE) 간의 직접적인 데이터 전달을 가능하게 하여 온칩 버퍼 의존도를 낮추고 병렬성을 극대화하였다.
2. **Hybrid, Hardware-friendly (H2) 양자화**: Vision Mamba의 데이터 분포 특성을 분석하여, 분산이 적은 가중치에는 Tensor-granularity 양자화를, 변동성이 큰 활성화 함수(activation)에는 Channel-granularity 양자화를 적용하는 하이브리드 방식을 도입하였다. 또한, 스케일링 인자를 2의 거듭제곱 형태로 근사하여 복잡한 곱셈 연산을 단순한 시프트(shift) 연산으로 대체하였다.

## 📎 Related Works

논문에서는 SSM 및 ViT 가속기와 관련된 기존 연구들을 다음과 같이 설명한다.

- **SSM 가속기**: VGA는 H3 모델의 FFT 연산을 가속하며, MARCA는 대규모 데이터센터 배포를 위해 HBM과 대용량 SRAM을 사용하는 Mamba 기반 LLM 가속기를 제안하였다. 반면, Mamba-X는 데이터센터가 아닌 **자원이 극도로 제한된 엣지 환경**과 **비전 모델**의 특성에 집중한다는 점에서 차별화된다.
- **ViT 가속기**: ViTCoD나 ViTALiTy 등은 Attention 메커니즘의 병목을 줄이기 위해 가지치기(pruning)나 저차원 근사(low-rank approximation)를 사용하였다. 하지만 Vision Mamba는 Attention 대신 Selective SSM을 사용하므로, 기존 ViT 가속 방식으로는 해결할 수 없는 새로운 형태의 순차적 의존성 문제가 존재한다.
- **양자화 연구**: 기존 PTQ(Post-Training Quantization) 기법들은 주로 ViT의 Attention 메커니즘에 집중해 왔다. Mamba-X는 Selective SSM 블록 내의 활성화 텐서가 갖는 높은 분산과 아웃라이어(outlier) 문제를 해결하기 위한 전용 하이브리드 양자화 전략을 제시한다.

## 🛠️ Methodology

### 1. 전체 시스템 아키텍처

Mamba-X는 Vision Mamba의 전체 파이프라인을 처리하기 위한 전용 하드웨어 유닛들로 구성된다.

- **DMA**: 온칩/오프칩 데이터 이동을 제어한다.
- **GEMM Engine**: Output-stationary 시스톨릭 어레이 기반으로 모든 선형 투영(linear projection) 연산을 수행한다.
- **VPU (Vector Processing Unit)**: LayerNorm, Conv1D, element-wise 연산을 처리한다.
- **SFU (Special Function Unit)**: SiLU, exponential, softplus와 같은 비선형 함수를 처리한다.
- **SSA (Systolic Scan Array)**: 본 논문의 핵심으로, Selective Scan 연산을 수행한다.
- **PPU (Post Processing Unit)**: Scan 이후의 MAC 연산 및 최종 출력을 처리하며, 청크 간 의존성을 해결하는 **LISU (Long Input Support Unit)**를 포함한다.

### 2. Systolic Scan Array (SSA) 및 데이터플로우

SSA는 Kogge-Stone 알고리즘의 데이터흐름을 시스톨릭 구조로 변환하여 구현되었다.

- **Chunk-wise Parallel Scan**: 입력 시퀀스를 $L$ 차원을 따라 여러 청크(chunk)로 나누어 병렬 처리한다.
- **SPE (Scan Processing Element)**: 각 SPE는 두 개의 곱셈기와 하나의 덧셈기를 포함하며, $\Delta A$와 $\Delta B \cdot u$ 데이터를 입력받아 상태 변수를 업데이트하고 이를 인접 SPE로 즉시 전달한다.
- **LISU (Long Input Support Unit)**: 청크 기반 처리 시 발생하는 청크 간 의존성(inter-chunk dependency)을 해결한다. 이전 청크의 최종 상태 변수를 저장하고 다음 청크의 계산에 즉시 제공하여 오프칩 메모리 접근 없이 연속적인 스캔을 가능하게 한다.

### 3. Special Function Unit (SFU)

비선형 함수를 효율적으로 계산하기 위해 **LUT(Look-Up Table) 기반의 piecewise linear interpolation** 방식을 사용한다.

- 입력 분포를 분석하여 99.9%의 데이터가 집중되는 구간을 설정하고, 해당 구간을 16~32개의 세그먼트로 나누어 선형 함수 $y = ax + b$로 근사한다.
- **ADU (Address Decoding Unit)** $\rightarrow$ **LUT** $\rightarrow$ **CU (Compute Unit)** 순으로 데이터를 처리하여 연산 비용을 획기적으로 줄였다.

### 4. Hybrid, Hardware-friendly (H2) Quantization

메모리 사용량을 줄이고 연산 효율을 높이기 위해 INT8 양자화를 적용한다.

- **Hybrid Strategy**:
  - **Weights**: 분포가 균일하므로 Tensor-granularity 양자화 적용.
  - **Activations**: 채널 간 편차가 크므로 Channel-granularity 양자화 적용.
- **Scaling Factor Approximation**:
    양자화된 값들의 연산 과정에서 발생하는 리스케일링(rescaling) 곱셈 연산을 줄이기 위해, 스케일링 인자를 가장 가까운 2의 거듭제곱($2^n$)으로 반올림한다. 이를 통해 하드웨어에서 곱셈기 대신 **시프트(shift) 연산**만으로 리스케일링을 수행할 수 있어 면적과 에너지를 절감한다.

## 📊 Results

### 1. 실험 설정

- **비교 대상**: NVIDIA Jetson AGX Xavier (Edge GPU).
- **대상 모델**: Vision Mamba Tiny, Small, Base.
- **측정 지표**: 처리량(Throughput), 지연 시간(Latency), 에너지 효율, 면적 효율, ImageNet-1K Top-1/5 정확도.

### 2. 주요 결과

- **Selective Scan 성능**: Mamba-X는 baseline GPU 대비 Selective Scan 처리량에서 평균 **$11.6\times$ 향상**을 보였다.
- **에너지 및 면적 효율**:
  - Selective Scan 실행 시 에너지 효율이 평균 **$11.5\times$ 향상**되었다.
  - 단위 면적당 성능(Performance/Area)은 **$601\times$ 증가**하였다. 이는 Mamba-X의 전체 면적($1.34\text{ mm}^2$ at 12nm)이 Jetson AGX Xavier($350\text{ mm}^2$)에 비해 매우 작기 때문이다.
- **End-to-End 성능**: Selective Scan의 병목을 해결함으로써 전체 추론 지연 시간을 평균 **$2.3\times$ 단축**시켰다.
- **정확도**: FP16 baseline 대비 Top-1 정확도 손실이 **1%p 미만**으로 유지되어, 양자화 및 근사 기법이 모델 성능에 미치는 영향이 매우 적음을 입증하였다.

## 🧠 Insights & Discussion

본 연구의 강점은 Vision Mamba의 성능 병목이 단순히 연산량의 문제가 아니라 **데이터 의존성에 따른 하드웨어 활용도 저하**와 **메모리 계층 구조의 한계**에 있음을 정확히 짚어낸 점이다. 이를 위해 알고리즘(Kogge-Stone)을 하드웨어 구조(Systolic Array)에 맞게 재설계한 SSA와, 데이터 분포 특성을 반영한 H2 양자화라는 두 가지 핵심 전략을 유기적으로 결합하였다.

다만, 실험 결과에서 알 수 있듯이 모델의 크기가 커질수록(Base 모델 등) 전체 지연 시간에서 GEMM 연산이 차지하는 비중이 상대적으로 높아지기 때문에, SSA를 통한 Selective Scan 가속만으로는 End-to-End 성능 향상 폭이 점차 줄어드는 경향이 있다. 이는 향후 연구에서 SSM 가속뿐만 아니라 엣지 환경에 최적화된 GEMM 엔진과의 통합 최적화가 추가로 필요함을 시사한다.

또한, 양자화 과정에서 사용된 캘리브레이션 데이터셋(ImageNet-1K의 1%)이 실제 추론 환경의 모든 데이터 분포를 대표할 수 있는지에 대한 일반화 가능성 문제가 남아 있으나, 본 논문에서는 1% 미만의 정확도 하락으로 이를 충분히 방어하였다고 판단한다.

## 📌 TL;DR

Mamba-X는 엣지 장치에서 Vision Mamba의 치명적인 병목인 Selective Scan의 순차적 의존성과 메모리 트래픽 문제를 해결하는 전용 가속기이다. **Kogge-Stone 알고리즘 기반의 Systolic Scan Array(SSA)**와 **채널별 하이브리드 양자화(H2 Quantization)**를 도입하여, GPU 대비 Selective Scan 처리량을 $11.6\times$ 높였으며, 면적 효율을 $601\times$ 극대화하였다. 이 연구는 고해상도 비전 태스크를 위한 SSM 기반 모델의 실질적인 온디바이스 AI 구현 가능성을 제시하였다는 점에서 매우 중요한 의미를 갖는다.
