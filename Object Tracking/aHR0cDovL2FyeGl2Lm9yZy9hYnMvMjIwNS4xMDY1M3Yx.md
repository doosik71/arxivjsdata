# Towards real-time and energy efficient Siamese tracking – a hardware-software approach

Dominika Przewlocka-Rus and Tomasz Kryjak (2022)

## 🧩 Problem to Solve

본 논문은 Visual Object Tracking (VOT) 분야에서 최신 성능을 보이는 Siamese tracker들이 가진 높은 계산 복잡도와 에너지 소모 문제를 해결하고자 한다. 일반적으로 Siamese tracker들은 높은 정확도를 달성하기 위해 대규모 병렬 처리가 가능한 고성능 GPU에서 구동되는데, 이는 에너지 효율성이 낮아 전력 소모에 민감한 임베디드 시스템이나 엣지 디바이스(예: 드론, 자율주행 자동차의 ADAS)에 적용하기 어렵다는 한계가 있다.

따라서 본 연구의 목표는 ARM 프로세서와 프로그래밍 가능한 로직(FPGA)이 결합된 이기종 플랫폼인 SoC FPGA를 활용하여, 실시간 처리 속도를 유지하면서도 에너지 효율적인 Siamese tracker의 하드웨어-소프트웨어 구현 방법을 제시하는 것이다. 특히, 잘 알려진 SiamFC 알고리즘을 기반으로 하여 정확도 손실을 최소화하면서 하드웨어 자원 사용량을 최적화하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

첫째, Zynq UltraScale+ MPSoC ZCU104 플랫폼 상에서 Siamese tracker를 위한 하드웨어-소프트웨어 통합 구현을 수행하였다. 이를 통해 알고리즘의 각 구성 요소별 시간 분석을 수행하고, 하드웨어 자원 사용량(에너지)과 처리 속도(FPS) 사이의 관계를 분석하는 Design Space Exploration(DSE)을 진행하였다.

둘째, FINN 가속기(accelerator)에 최적화된 알고리즘-가속기 공동 설계(Algorithm-accelerator co-design)를 통해 양자화된(quantized) Siamese 네트워크 아키텍처를 제안하였다. 이를 통해 기존 SiamFC와 유사한 수준의 추적 정확도를 유지하면서도 파라미터 수를 획기적으로 줄여 계산 효율성을 높였다.

## 📎 Related Works

Siamese tracker는 두 개의 입력(Exemplar image와 Region of Interest, ROI)의 유사도를 측정하는 Y자형 네트워크 구조를 가진다. 기존 연구들은 다음과 같이 발전해 왔다.

- **SiamFC**: AlexNet 기반의 Fully Convolutional 구조를 사용하여 크로스 상관관계(cross-correlation)를 통해 타겟의 위치를 찾는 기초적인 모델이다.
- **SiamRPN 및 SiamRPN++**: 단순한 유사도 맵을 넘어 Region Proposal Network(RPN)를 도입하여 바운딩 박스(bounding box)의 정밀도를 높이고, ResNet50과 같은 더 깊은 백본(backbone) 네트워크와 다중 레벨 특징 융합을 사용하여 성능을 개선하였다.
- **기타 개선**: VGG-16 백본 적용, depth-wise separable correlation 도입, 세그멘테이션 마스크 예측 등을 통해 정확도를 높이려는 시도가 계속되었다.

그러나 이러한 최신 알고리즘들은 대부분 고성능 GPU를 전제로 하며, 임베디드 디바이스로의 배포에 관한 연구는 부족한 실정이다. 기존의 FPGA 가속 연구들은 하드웨어 자원 사용량이나 에너지 소모, 혹은 소프트웨어 베이스라인 대비 정확도 비교 등의 상세 정보가 누락되어 있어 객관적인 비교가 어려웠다는 한계가 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 파이프라인
본 시스템은 하드웨어-소프트웨어 협력 설계 방식을 채택하여 역할을 분담한다.
- **PS (Processor System, ARM)**: Python 스크립트를 통해 구동되며, 입력/출력 처리, FPGA 가속기와의 통신, 전처리(ROI 크롭 및 스케일링), 후처리(유사도 맵 기반 타겟 위치 결정)를 담당한다.
- **PL (Programmable Logic, FPGA)**: FINN 프레임워크를 사용하여 구현된 양자화된 Siamese 네트워크가 배치되어 추론(Inference) 과정을 가속화한다.

### 2. 제안하는 네트워크 아키텍처 및 양자화
FINN 가속기의 제약 사항(제한된 온보드 자원, 연산 정밀도)을 고려하여 맞춤형 Siamese 네트워크를 설계하였다.
- **구조**: 총 5개의 Convolution 레이어로 구성되며, 모든 레이어는 $3 \times 3$ 커널을 사용한다. 각 레이어 이후에는 Batch Normalization이 적용된다 (마지막 레이어 제외).
- **입력 크기**: ROI는 $238 \times 238 \times 3$, 추적 대상(object) 이미지는 $110 \times 110 \times 3$ 크기를 가진다.
- **양자화(Quantization)**: 
    - 활성화 함수(Activations)는 4-bit로 양자화하였다.
    - 가중치(Weights)는 첫 번째 레이어와 마지막 레이어의 정확도 유지를 위해 8-bit 정밀도를 사용하고, 나머지 레이어는 4-bit를 적용하였다.

### 3. 하드웨어 최적화 및 가속화 전략
FINN 가속기에서 병렬 처리 수준을 조절하기 위해 Folding 파라미터를 조정하였다.
- **PE (Processing Element)**: 동시에 처리되는 입력 채널의 수를 결정한다.
- **SIMD (Single Instruction Multiple Data)**: 집계되는 출력 채널의 수를 결정한다.
이 두 파라미터를 조정함으로써 하드웨어 자원 사용량(LUT, BRAM 등)과 처리 속도(FPS) 사이의 trade-off를 분석하는 Design Space Exploration을 수행하였다.

## 📊 Results

### 1. 추적 정확도 평가
GOT 10k 데이터셋으로 학습시키고 VOT 2016 데이터셋에서 평가를 진행하였다. 성능 지표로는 mAO(mean Average Overlap)를 사용하였다.

- **정확도**: 3-scale 처리를 적용한 양자화 모델의 mAO는 $0.355$로, floating-point(FP32) 모델($0.362$) 및 오리지널 SiamFC($0.385$)와 유사한 수준의 성능을 보였다.
- **효율성**: 제안된 네트워크의 파라미터 수는 554,688개로, AlexNet 기반의 오리지널 SiamFC(3,747,200개)보다 약 6.7배 더 작다.
- **스케일 영향**: 3-scale에서 1-scale로 줄일 경우, 양자화 모델의 정확도는 $0.355$에서 $0.281$로 크게 하락하는 양상을 보였다.

### 2. 하드웨어 성능 분석
다양한 Folding 설정(V1~V6)에 따른 성능을 측정한 결과는 다음과 같다.

- **최적 설정 (V5)**: LUT 사용량 약 66.87%, BRAM 약 55.29%를 사용하여 네트워크 단독 처리 속도 약 49 FPS를 달성하였으며, 이때 에너지 소모는 $5.5\text{W}$였다.
- **한계 설정 (V6)**: 병렬성을 극도로 높였으나 LUT/BRAM 사용량이 90%를 초과하고 에너지는 $6.79\text{W}$까지 상승했지만, 속도 향상은 $0.6\text{FPS}$ 수준으로 미미하여 효율성이 급격히 떨어진다.

### 3. 전체 시스템 지연 시간 (End-to-End Latency)
V5 설정을 기준으로 전체 추적 시스템의 시간 분석을 수행한 결과, 전체 시스템 속도는 약 **17 FPS**로 측정되었다.

- **시간 소요 비중**:
    - FINN 네트워크 실행 및 데이터 전송: 약 52% (가장 큰 비중)
    - 네트워크 출력 후처리 (Cross-correlation 등): 약 25%
    - 입력 전처리 (Crop & Resize): 약 18%
- 특히 가속기에서 데이터를 가져오는 Output transfer 시간이 Input transfer보다 훨씬 더 많은 시간을 소모함을 확인하였다.

## 🧠 Insights & Discussion

본 연구를 통해 고성능 GPU 없이도 FPGA를 활용하여 에너지 효율적인 실시간 Siamese 추적이 가능함을 입증하였다. 오리지널 SiamFC가 NVIDIA GeForce GTX Titan X(250W)에서 83 FPS를 낸 것에 비해, 본 구현체는 단 $5.5\text{W}$의 전력으로 17 FPS를 달성하여 전력 대비 효율성을 극대화하였다.

하지만 다음과 같은 한계점과 개선 방향이 논의된다.
첫째, **정확도의 한계**이다. SiamFC와는 유사한 성능을 보이지만, 최신 SOTA 알고리즘(SiamRPN++ 등)에 비해 정확도가 낮다. 이는 FINN 가속기가 바운딩 박스 회귀(Bounding box regression)나 다중 레벨 특징 융합과 같은 복잡한 연산을 직접적으로 지원하지 않기 때문이다. 향후에는 FINN과 커스텀 하드웨어 모듈을 결합하여 이를 해결해야 한다.

둘째, **데이터 전송 및 후처리의 병목 현상**이다. 전체 시간의 약 40%가 PS와 PL 사이의 데이터 전송 및 소프트웨어 기반 후처리에 소요된다. 만약 Cross-correlation과 같은 후처리 과정을 FPGA 내부로 이동시켜 하드웨어로 가속한다면, PS로의 데이터 전송량을 줄이고 전체 FPS를 획기적으로 높일 수 있을 것이다.

## 📌 TL;DR

본 논문은 전력 소모가 큰 GPU 기반의 Siamese tracker를 대체하기 위해, **FINN 가속기와 Zynq UltraScale+ MPSoC FPGA를 이용한 에너지 효율적인 하드웨어-소프트웨어 구현**을 제안한다. 4-bit/8-bit 양자화된 맞춤형 네트워크를 설계하여 파라미터 수를 6.7배 줄이면서도 SiamFC 수준의 정확도를 유지하였으며, 최종적으로 **5.5W의 저전력으로 17 FPS의 실시간 추적 성능**을 확보하였다. 이 연구는 고성능 비전 알고리즘을 저전력 엣지 디바이스에 배포하기 위한 가속기 설계 및 최적화의 가이드라인을 제공한다는 점에서 중요한 의미를 갖는다.