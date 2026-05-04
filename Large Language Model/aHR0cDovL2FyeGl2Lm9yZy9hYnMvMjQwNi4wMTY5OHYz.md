# Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models

Abhimanyu Rajeshkumar Bambhaniya, Ritik Raj, Geonhwa Jeong, Souvik Kundu, Sudarshan Srinivasan, Suvinay Subramanian, Midhilesh Elavazhagan, Madhu Kumar, Tushar Krishna (2025)

## 🧩 Problem to Solve

최근 Large Language Model(LLM)은 비약적인 성능 향상을 이루었으나, 이를 실제 서비스에 배포하기 위해서는 방대한 연산량, 메모리, 네트워크 자원을 갖춘 하드웨어 플랫폼 설계가 필수적이다. 특히 LLM의 추론(Inference)은 일반적인 딥러닝 모델과 달리 Prefill(입력 토큰 처리)과 Decode(출력 토큰 생성)라는 두 가지 서로 다른 단계로 나뉘며, 각각 연산 집약적(Compute-bound) 특성과 메모리 집약적(Memory-bound) 특성을 동시에 가진다.

현재 LLM 아키텍처와 서빙 최적화 기법(Quantization, Paged Attention, Speculative Decoding 등)이 매우 빠르게 발전하고 있음에도 불구하고, 특정 서비스 수준 목표(Service Level Objectives, SLO)를 달성하기 위해 어떤 하드웨어 사양(Compute, Memory, Network)이 필요한지에 대한 정량적인 분석 도구가 부족하다. 따라서 본 논문은 다양한 LLM 아키텍처, 최적화 기법, 그리고 하드웨어 플랫폼 파라미터 간의 관계를 분석하여 최적의 AI 플랫폼 설계를 가이드하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **GenZ (Generative LLM analyZer)**라는 분석 프레임워크를 제안한 것이다. GenZ는 단순한 NPU 시뮬레이터가 아니라, 모델 아키텍처, 소프트웨어 최적화, 분산 하드웨어 플랫폼이라는 세 가지 축을 통합적으로 모델링하여 엔드-투-엔드 추론 성능을 예측한다.

중심적인 설계 아이디어는 모델의 연산자(Operator) 수준의 프로파일링, NPU의 하드웨어 특성(Efficiency Factor), 그리고 다차원 네트워크 토폴로지를 계층적으로 결합하여, 실제 하드웨어 없이도 다양한 설계 공간 탐색(Design Space Exploration, DSE)이 가능하게 하는 것이다.

## 📎 Related Works

기존의 LLM 추론 분석 도구들은 다음과 같은 한계를 가진다.
- **ASTRA-sim, MadMax, vTrain**: 주로 분산 학습(Distributed Training)의 통신 최적화에 집중하며, 다양한 LLM 아키텍처나 추론 전용 최적화 기법을 모델링하지 않는다.
- **LLM-Viewer**: Roofline 모델 기반의 이상적인 성능 추정치를 제공하지만, 상세한 시스템 최적화나 분산 플랫폼의 복잡성을 반영하기 어렵다.
- **Vidur, LLMServingSim**: 추론 시스템의 스케줄링 기법에 집중하며, 주로 기존 하드웨어 사양 내에서의 분석에 머문다.
- **LLM-Compass**: ASIC 설정을 위한 DSE를 수행하지만, 주로 Dense 모델에 한정된다.

**GenZ와의 차별점**: GenZ는 MoE(Mixture of Experts), Mamba와 같은 최신 아키텍처는 물론, Speculative Decoding, KV Pruning, Mixed Precision 등 광범위한 최적화 기법을 모두 지원하며, 이를 다차원 네트워크 토폴로지와 결합하여 분석할 수 있는 최초의 프레임워크이다.

## 🛠️ Methodology

GenZ는 크게 세 가지 주요 구성 요소로 이루어져 있다.

### 1. Model Profiler
모델의 구조적 특성을 분석하는 단계이다. HuggingFace의 `AutoModels`를 사용하여 각 연산자의 형태(Shape)를 결정하고, 이를 통해 전체 연산량, 메모리 점유율, KV Cache 크기, 통신량(Collective sizes)을 계산한다. 특히 Dense 모델뿐만 아니라 MoE, Mamba, GQA(Grouped-Query Attention) 등 다양한 아키텍처를 지원하며, Quantization 및 Chunked Prefill과 같은 최적화 기법을 적용한 상태의 연산 그래프를 생성한다.

### 2. NPU Characterizer
개별 가속기(NPU)의 성능을 모델링한다. 각 NPU는 연산 능력($FLOPS$), 고속 메모리 대역폭($BW_{mem}$), 저속 메모리(Offload용) 대역폭($BW_{omem}$) 및 용량을 가진다. 

연산자의 실행 시간 $T_{op}$는 Roofline 모델을 기반으로 다음과 같이 계산한다:
$$T_{op} = \max\left(\frac{C_{op}}{FLOPS \times Eff_C}, \frac{M_{op}}{BW_{mem} \times Eff_{mem}}\right)$$
여기서 $C_{op}$는 연산 횟수, $M_{op}$는 메모리 접근량이며, $Eff_C$와 $Eff_{mem}$은 소프트웨어 오버헤드와 동기화 문제를 반영한 효율성 계수(Efficiency Factor)이다.

### 3. Platform Characterizer
여러 NPU가 연결된 분산 플랫폼을 모델링한다.
- **네트워크 토폴로지**: 다차원 상호연결망(ICN)의 링크 지연 시간($T_{link}$), 대역폭($BW_{link}$), 효율성($Eff_{link}$)을 정의한다.
- **병렬화 전략**: Tensor Parallelism(TP), Pipeline Parallelism(PP), Expert Parallelism(EP), Sequence Parallelism(SP)을 지원하며, 이를 물리적 토폴로지에 매핑한다.
- **집단 통신(Collectives)**: AllReduce, All-to-All, Send-Recv 등의 통신 패턴을 생성하며, 상세한 통신 시간 예측을 위해 ASTRA-sim의 시스템 레이어를 활용한다.

## 📊 Results

### 검증 (Validation)
GenZ는 NVIDIA H100, A100, Intel Gaudi2, AMD MI300x, SambaNova SN40L 등 5가지 실제 하드웨어 플랫폼에서 검증되었다. 다양한 모델과 워크로드에 대해 실제 측정값과 GenZ 예측값 사이의 기하평균 오차(Geomean Error)는 **5.82%**로 나타나, 매우 높은 예측 정확도를 보였다.

### 주요 사례 연구 결과
1. **하드웨어 특성 스케일링**:
   - **TFLOPS 증가**: 긴 컨텍스트의 Prefill 단계에서 성능 향상이 뚜렷하나, 메모리 집약적인 Decode 단계에서는 효과가 거의 없다.
   - **Memory BW 증가**: Decode 단계의 지연 시간이 대역폭 증가에 비례하여 감소한다.
   - **ICN BW 및 Latency**: Prefill은 대용량 메시지를 주고받으므로 대역폭(BW)에 민감하고, Decode는 소량의 메시지를 자주 주고받으므로 링크 지연 시간(Latency)에 매우 민감하다.

2. **플랫폼 아키텍처 비교**:
   - **SRAM Wafer (e.g., Cerebras)**: 모델이 SRAM에 모두 들어갈 경우, 압도적인 메모리 대역폭으로 인해 Prefill 및 Decode에서 최고의 성능과 에너지 효율을 보인다.
   - **Multiple GPUs (e.g., NVIDIA)**: 모델이 SRAM에 들어가지 않는 대규모 모델의 경우, 높은 집계 메모리 대역폭을 바탕으로 Decode 및 Chunked 워크로드에서 효율적이다.
   - **Transformer ASICs (e.g., Etched)**: 매우 큰 모델이나 매우 긴 컨텍스트 환경에서 높은 연산 능력을 바탕으로 우수한 성능을 보인다.

3. **AI Assistant를 위한 극한 요구사항**:
   10T 파라미터 규모의 Super-LLM을 실시간 개인 비서로 구현하기 위해 분석한 결과, 메모리 대역폭보다 **메모리 용량(Capacity)**이 더 심각한 병목 지점임이 밝혀졌다. 2M 토큰 컨텍스트를 지원하려면 약 15TB의 메모리가 필요하며, 이는 약 400개의 HBM3e 스택에 해당하는 양으로 현재 기술로는 지속 가능성이 낮다.

## 🧠 Insights & Discussion

본 논문은 LLM 추론의 병목 지점이 모델 아키텍처와 최적화 기법에 따라 동적으로 변화한다는 점을 시사한다.
- **GQA의 영향**: Dense 모델은 Chunked Prefill 시 KV Cache로 인한 메모리 대역폭 병목이 심하지만, GQA 모델은 이 영향이 적어 오히려 연산 능력(Compute)이 주 병목이 된다.
- **Speculative Decoding의 트레이드오프**: 토큰 생성 속도를 높일 수 있으나, Draft 모델을 위한 추가 메모리 용량이 필요하며, 타겟 모델에 더 많은 토큰을 한 번에 입력하게 되어 연산 집약적인 특성이 강해진다.
- **MoE 병렬화**: Prefill 단계에서는 전문가(Expert)들이 균등하게 활성화된다는 가정하에 Expert Parallelism(EP)이 유리하지만, Decode 단계에서는 적은 수의 토큰만 라우팅되므로 TP 또는 TP+EP 조합이 더 효율적이다.

비판적으로 보자면, 본 연구는 에너지 소비 모델을 단순 선형 함수로 가정하였으며, 저수준의 마이크로아키텍처 세부 사항을 효율성 계수($Eff$)라는 상수로 추상화하였다. 따라서 실제 하드웨어의 동적인 전력 관리나 복잡한 캐시 계층 구조에 의한 성능 변동을 완벽하게 포착하지 못했을 가능성이 있다.

## 📌 TL;DR

본 논문은 LLM 추론 성능을 예측하기 위한 분석 프레임워크 **GenZ**를 제안한다. GenZ는 모델 아키텍처, 소프트웨어 최적화, 분산 하드웨어 설계를 통합적으로 모델링하여 실제 하드웨어와 5.82%의 오차 범위 내에서 성능을 예측한다. 연구 결과, 차세대 초거대 모델(10T+)의 실시간 서비스 구현을 위해서는 메모리 대역폭보다 **메모리 용량의 확보**가 가장 시급한 과제임을 정량적으로 제시하였다. 이 도구는 향후 AI 가속기 설계자와 시스템 아키텍트가 특정 SLO를 만족하는 최적의 하드웨어 사양을 결정하는 데 중요한 역할을 할 것으로 기대된다.