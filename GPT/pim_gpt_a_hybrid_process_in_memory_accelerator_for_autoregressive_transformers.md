# PIM-GPT: A Hybrid Process-in-Memory Accelerator for Autoregressive Transformers

Yuting Wu, Ziyu Wang, Wei D. Lu (2024)

## 🧩 Problem to Solve

본 논문은 GPT와 같은 Decoder-only Transformer 모델의 추론 과정에서 발생하는 메모리 병목 현상을 해결하고자 한다. GPT 모델은 다음 토큰을 순차적으로 예측하는 autoregressive(자기회귀) 방식으로 동작하는데, 이 과정에서 다음과 같은 하드웨어적 한계가 발생한다.

첫째, GPT는 모델 크기가 매우 크지만, 연산량 대비 메모리 접근 횟수가 많은 '낮은 연산-메모리 비율(low compute-to-memory ratio)' 특성을 가진다. 둘째, GPU와 같은 기존 하드웨어는 Matrix-Matrix Multiplication(MMM)에 최적화되어 있으나, GPT의 토큰 생성 단계에서는 Vector-Matrix Multiplication(VMM)이 주를 이룬다. 이 VMM 연산은 데이터 재사용률이 극도로 낮아, 대량의 가중치 데이터를 외부 메모리에서 계속해서 읽어와야 하며, 이는 GPU의 강력한 연산 유닛들을 제대로 활용하지 못하게 하고 막대한 에너지 소모를 야기한다.

따라서 본 연구의 목표는 데이터 이동을 최소화하고 내부 대역폭을 극대화할 수 있는 Process-in-Memory(PIM) 아키텍처를 통해 GPT 추론의 성능과 에너지 효율을 끝단-끝단(end-to-end)으로 가속화하는 PIM-GPT 시스템을 설계하는 것이다.

## ✨ Key Contributions

PIM-GPT의 핵심 아이디어는 **'연산 특성에 따른 하이브리드 분리'**와 **'하드웨어 인식 매핑(Hardware-aware Mapping)'**이다.

1. **하이브리드 가속 구조**: 데이터 집약적인 VMM 연산은 DRAM 내부의 PIM 유닛에서 직접 처리하여 off-chip 데이터 이동을 제거하고, PIM에서 구현하기 비용이 많이 드는 비선형 함수 및 데이터 통신은 전용 ASIC(Application Specific Integrated Circuit)에서 처리하는 구조를 제안한다.
2. **최적화된 데이터 매핑**: DRAM의 물리적 구조(Channel, Bank)를 고려하여 가중치 행렬을 분할하고 배치함으로써, 데이터 지역성(Data Locality)을 극대화하고 연산 병렬성을 높이는 매핑 스킴을 설계하였다.
3. **실용적인 end-to-end 가속**: 기존 연구들이 주로 Attention 메커니즘이나 특정 레이어에만 집중한 것과 달리, Feed-Forward Network(FFN)를 포함한 전체 추론 파이프라인을 가속하며, 고가의 HBM(High Bandwidth Memory) 없이 일반적인 GDDR6 기반 PIM으로 구현 가능한 실용적인 솔루션을 제시한다.

## 📎 Related Works

기존의 Transformer 가속기들은 다음과 같은 한계를 가지고 있다.

- **하드웨어 오버헤드**: HBM과 같은 고가의 메모리나 복잡한 in-memory logic을 사용하여 구현 비용이 높다.
- **모델 변형 필요**: 성능 향상을 위해 모델 프루닝(pruning)이나 양자화(quantization)를 적용하는데, 이는 모델의 유연성을 떨어뜨리고 추론 정확도 손실을 초래한다.
- **부분적 가속**: 많은 연구가 Attention 계산에만 집중하며, LayerNorm이나 Residual connection 같은 end-to-end 구성 요소들을 간과한다.
- **모델 대상의 차이**: BERT와 같은 Encoder-only 모델을 위한 가속기가 많으나, GPT와 같은 Decoder-only 모델의 autoregressive 특성(메모리 바운드 문제)을 해결하는 설계는 부족했다.

PIM-GPT는 DRAM 뱅크 수준에서 MAC(Multiply-Accumulate) 유닛을 배치하는 PNM(Process-Near-Memory) 방식을 채택하여, 구현 가능성과 성능 사이의 균형을 맞춤으로써 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

PIM-GPT는 **DRAM 기반 PIM 칩**과 **커스텀 ASIC**으로 구성된 하이브리드 시스템이다.

- **PIM Chips**: 가중치 행렬을 저장하고, 입력 벡터를 브로드캐스팅하여 VMM 연산을 수행한다.
- **ASIC**: LayerNorm, Softmax, GELU와 같은 비선형 연산, 데이터 통신 및 PIM 칩 제어를 담당한다.
- **데이터 포맷**: 성능과 정확도의 균형을 위해 BFloat16(BF16) 형식을 사용한다.

### DRAM 기반 PIM 및 VMM 연산

PIM-GPT는 각 DRAM 뱅크마다 MAC 유닛을 배치한다. 8개의 채널이 있으며, 각 채널은 16개의 뱅크로 구성되어 총 128개의 뱅크가 동시에 MAC 연산을 수행할 수 있다.

- **동작 과정**: ASIC으로부터 입력 벡터가 전달되면 PIM 내의 SRAM 버퍼에 저장되고, 모든 뱅크의 MAC 유닛으로 브로드캐스팅된다. 각 뱅크는 자신의 로컬 메모리에 저장된 가중치와 벡터를 곱하여 부분 합(partial sum)을 계산한다.
- **최적화**: 행 활성화(Row Activation) 비용을 줄이기 위해 'Open-row policy'를 사용하며, 행 버퍼(row buffer)에서 데이터를 최대한 소비하도록 매핑한다.

### ASIC 아키텍처 및 근사 연산

비선형 함수를 효율적으로 처리하기 위해 ASIC에서는 다음과 같은 근사 알고리즘을 사용한다.

- **Softmax 및 GELU**: 테일러 급수(Taylor series) 근사를 통해 덧셈과 곱셈만으로 계산한다.
- **나눗셈(Division)**: Newton-Raphson 방법을 사용하여 역수를 구한 뒤 곱하는 방식으로 처리한다.
- **제곱근 역수(Inverse Square Root)**: Quake III Arena의 fast inverse square root 알고리즘을 채택하여 빠르게 계산한다.

### 데이터 매핑 전략 (Mapping Scheme)

데이터 지역성을 높이고 병렬성을 극대화하기 위해 세 가지 전략을 사용한다.

1. **Attention Head Mapping**: 여러 개의 Attention head를 하나로 결합(Concatenate)하여 DRAM 행(row) 크기에 맞게 채운다. 이를 통해 Row Hit Rate를 높이고 ACT/PRE 명령 횟수를 줄인다.
2. **Intermediate Data Reservation**: 생성되는 Key, Value 벡터를 저장하기 위한 공간을 미리 예약한다. Key 행렬은 이후 전치(transpose) 연산이 필요하므로 Row-major로, Value 행렬은 Column-major로 저장한다.
3. **Weight Matrix Tiling**: 가중치 행렬의 크기가 뱅크 행 크기를 초과할 경우, 행렬과 벡터를 청크(chunk) 단위로 나누어 처리하고 그 결과(partial VMM results)를 ASIC에서 합산한다.

## 📊 Results

### 실험 설정

- **비교 대상**: NVIDIA T4 GPU, Intel Xeon Gold 6154 CPU.
- **테스트 모델**: 최대 14억 개의 파라미터를 가진 8종의 GPT2 및 GPT3 모델.
- **측정 지표**: 추론 지연 시간(Latency), 에너지 효율, Row Hit Rate.
- **하드웨어 사양**: TSMC 28nm 공정 기반 ASIC, GDDR6 기반 PIM 프로토타입 사양 적용.

### 주요 결과

- **속도 향상(Speedup)**: GPU 대비 $41 \sim 137\times$, CPU 대비 $631 \sim 1074\times$의 속도 향상을 달성하였다.
- **에너지 효율**: GPU 대비 $123 \sim 383\times$, CPU 대비 $320 \sim 602\times$의 에너지 효율 개선을 보였다.
- **데이터 이동 감소**: 가중치 데이터를 칩 내부에서 처리함으로써, 외부 I/O 데이터 전송량을 $110 \sim 259\times$ (GDDR6 기준) 감소시켰다.
- **효율성 지표**: 매핑 전략을 통해 약 $98\%$의 매우 높은 Row Hit Rate를 달성하였으며, 채널 수를 늘림에 따라 성능이 거의 선형적으로 증가하는 확장성을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

PIM-GPT의 성능 향상은 단순히 연산 속도를 높인 것이 아니라, GPT 추론의 근본적인 병목인 **'메모리 벽(Memory Wall)'**을 제거했기 때문에 가능했다. 특히 VMM 연산이 전체 지연 시간의 대부분을 차지하며(GPT3-XL의 경우 나머지 연산은 1.16%에 불과), 이 부분을 PIM으로 옮긴 것이 결정적이었다. 또한, ASIC의 동작 주파수를 1GHz에서 100MHz로 10배 낮추더라도 전체 성능 저하는 최대 20%에 그쳤는데, 이는 시스템 전체 성능이 ASIC의 연산 속도보다 PIM의 메모리 대역폭 및 데이터 로컬리티에 훨씬 더 의존적임을 시사한다.

### 한계 및 논의사항

본 논문에서는 비선형 함수(Softmax, GELU 등)를 위해 테일러 급수 및 Newton-Raphson 등의 근사 알고리즘을 사용하였다. 하드웨어 효율성은 극대화되었으나, 이러한 근사가 실제 모델의 최종 출력 정확도(Accuracy)에 미치는 영향에 대한 정량적 분석은 본문에 명시되지 않았다. 또한, 매우 큰 모델의 경우 ASIC에서 부분 합을 처리하는 오버헤드가 증가할 수 있으나, 여전히 GPU보다는 훨씬 효율적이라는 점을 강조하고 있다.

## 📌 TL;DR

PIM-GPT는 autoregressive Transformer(GPT)의 메모리 병목 문제를 해결하기 위해 **VMM 연산은 DRAM 내부(PIM)에서, 비선형 연산은 전용 ASIC에서 처리하는 하이브리드 가속기**이다. 효율적인 데이터 매핑 전략을 통해 데이터 이동을 획기적으로 줄임으로써, GPU 대비 최대 $137\times$의 속도 향상과 $383\times$의 에너지 효율을 달성하였다. 이 연구는 고가의 HBM 없이도 실용적인 PIM 설계를 통해 거대 언어 모델의 추론 성능을 극대화할 수 있음을 보여주며, 향후 온디바이스 AI나 대규모 LLM 추론 시스템 설계에 중요한 방향성을 제시한다.
