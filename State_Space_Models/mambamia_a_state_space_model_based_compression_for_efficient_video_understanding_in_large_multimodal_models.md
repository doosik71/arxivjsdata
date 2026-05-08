# MambaMia: A State-Space-Model-Based Compression for Efficient Video Understanding in Large Multimodal Models

Geewook Kim, Minjoon Seo (2025)

## 🧩 Problem to Solve

Large Multimodal Models(LMMs)는 시각적 입력을 통합하여 대규모 언어 모델(LLMs)의 능력을 확장하였으나, 다중 프레임 비디오를 처리할 때 심각한 메모리 및 계산 오버헤드가 발생하는 문제가 있다. 특히 고밀도(dense) 비디오나 긴 비디오의 경우, 생성되는 토큰의 수가 급격히 증가하는 '토큰 폭발(token explosion)' 현상이 발생하며, 이는 모델의 학습과 추론 단계 모두에 큰 부담을 준다.

기존의 해결책으로는 공간적 풀링(spatial pooling), 희소 샘플링(sparse sampling), 토큰 병합(token merging) 등이 제안되었으나, 이러한 방법들은 정보 손실이 크거나 특정 사용자 쿼리에 의존적(query-aware)이어서 범용성이 떨어진다는 한계가 있다. 따라서 본 논문은 리소스가 제한된 환경에서도 효율적으로 비디오의 광범위한 컨텍스트를 보존하면서 토큰 수를 획기적으로 줄일 수 있는 범용적인 압축 프레임워크의 필요성을 제기한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 State-Space Model(SSM)의 선형 계산 복잡도를 활용하여 시공간적 토큰을 효율적으로 압축하는 것이다. 이를 위해 다음과 같은 두 가지 주요 설계를 제안한다.

1. **TSTAR (Two-Stage Token Aggregation and Reduction) 프레임워크**: 초기 단계에서 고밀도로 프레임을 샘플링하여 세부 이벤트를 보존하고, 이후 신경망 아키텍처를 통해 토큰을 압축한 뒤, 마지막으로 2차 템포럴 다운샘플링 필터를 적용하는 계층적 구조이다.
2. **MambaMia 아키텍처**: Mamba 계열의 SSM을 기반으로 한 새로운 압축 레이어이다. Bidirectional Mamba 블록에 Gated Skip Connection과 학습 가능한 가중 평균 풀링(weighted-average pooling) 메커니즘을 결합하여, 로컬 시각 특징을 매우 압축된 비디오 표현으로 효율적으로 통합한다.

## 📎 Related Works

기존의 비디오 토큰 감소 방식은 크게 세 가지 방향으로 나뉜다.

- **공간적 토큰 감소**: Bilinear interpolation이나 average pooling 같은 단순 풀링, 또는 Q-Former와 같은 경량 어텐션 모듈을 사용하여 프레임당 토큰을 줄인다. 하지만 이를 단순히 프레임별로 적용하면 고밀도 비디오에서는 여전히 전체 토큰 수가 너무 많다는 문제가 있다.
- **시공간적 토큰 압축**: 3D 풀링이나 유사도 기반의 가지치기(pruning), 또는 학습된 쿼리를 사용하는 어텐션 기반 리샘플링 등이 있다. 그러나 이러한 방식들은 수천 개의 토큰을 여전히 필요로 하거나, 특정 쿼리에 종속되어 유연성이 떨어진다.
- **State-Space Models (SSMs)**: Mamba와 같은 SSM은 Transformer의 이차 복잡도($O(T^2)$)와 달리 선형 복잡도($O(T)$)를 가져 긴 시퀀스 처리에 매우 유리하다.

본 논문은 이러한 기존 방식들과 달리, 쿼리에 독립적(query-agnostic)이면서도 학습 가능한 방식을 통해 효율성과 정보 보존 사이의 균형을 맞춘 일반 목적의 압축 방식을 제안한다.

## 🛠️ Methodology

### 전체 파이프라인: TSTAR

TSTAR는 비전 백본과 LLM 사이에 위치하며 다음과 같은 단계로 동작한다.

1. **초기 고밀도 프레임 샘플링**: 정보 손실을 최소화하기 위해 일단 많은 수의 프레임을 샘플링한다.
2. **학습 가능한 쿼리 삽입**: 구조적 앵커 포인트 역할을 하는 학습 가능한 쿼리 토큰을 일정 간격($k$ 패치마다)으로 삽입한다.
3. **MambaMia 압축**: 삽입된 쿼리와 패치 토큰들이 MambaMia 레이어를 통과하며 시공간적 컨텍스트가 압축된다.
4. **2차 토큰 샘플링**: LLM에 입력하기 전, 압축된 토큰들 중 일부를 다시 샘플링하여 최종 토큰 예산을 맞춘다.

### MambaMia 블록 구조

MambaMia는 Bi-Mamba를 기반으로 하며, 여기에 **Gated Patch Aggregation** 모듈을 추가하여 정보를 효율적으로 통합한다.

**1. 가중 평균 풀링 (Weighted-Average Pooling)**
쿼리 토큰 $q \in \mathbb{R}^d$와 주변 패치 임베딩 $\{x_i\}_{i=1}^k$가 주어졌을 때, 선형 레이어와 softmax를 통해 가중치 $\alpha$를 계산하고 이를 이용해 주변 정보를 합산한다.
$$\alpha = \text{softmax}(W_\alpha q + b_\alpha), \quad a = \sum_{i=1}^k \alpha_i x_i$$

**2. 적응형 게이팅 메커니즘 (Adaptive Gating)**
계산된 합산 값 $a$를 원래의 쿼리 표현 $q$와 얼마나 섞을지를 결정하는 스칼라 게이트 $g \in [0, 1]$를 sigmoid 함수를 통해 생성한다.
$$g = \sigma(W_g q + b_g), \quad q_{\text{new}} = (1-g)q + ga$$
여기서 $g \approx 0$이면 기존 쿼리 컨텍스트를 보존하고, $g \approx 1$이면 주변의 로컬 정보를 강하게 반영한다.

### 학습 전략

본 논문은 두 가지 학습 패러다임을 탐색한다.

- **Unified Training**: 이미지와 비디오 모달리티를 한 단계에서 동시에 LLM에 통합하여 학습한다.
- **Two-Stage Training**: 먼저 이미지 수준의 명령 튜닝(instruction tuning)을 수행하여 LLM을 적응시킨 후, 비디오 작업으로 미세 조정(fine-tuning)한다. 실험 결과, 대규모 모델에서는 이 방식이 더 안정적인 성능을 보였다.

## 📊 Results

### 실험 설정

- **백본**: CLIP-ConvNeXt-Large (비전 인코더), Phi-3, Vicuna-7B, Vicuna-13B (LLM).
- **벤치마크**: MLVU, VideoMME, TempCompass, VNBench (VNBC/VNBI), LVBench 등 긴 비디오 이해 능력을 측정하는 지표 사용.
- **비교 대상**: No Compression, Spatial Pooling, C-Abstractor, Attention Resamplers, Bi-Mamba 등.

### 주요 결과

- **토큰 효율성**: TSTAR-MambaMia는 256프레임의 비디오를 처리할 때 단 860개의 토큰만을 사용하며, 이는 기존 방식들(수천~수만 개 토큰 사용)보다 훨씬 적은 수치이다.
- **성능**: 13B 규모의 모델은 VNBench에서 45.2점을 기록하며, 매우 적은 토큰 예산으로도 GPT-4V(48.9점)의 성능에 근접하는 결과를 보였다.
- **아키텍처 비교**: MambaMia 블록을 일반 Transformer 블록으로 교체했을 때 성능이 크게 저하됨을 확인하였으며, 이는 비디오 데이터 압축에 있어 SSM의 우수성을 입증한다.
- **추론 비용**: 프레임 수가 증가함에 따라 추론 지연 시간(Latency)과 GPU 메모리 사용량이 Transformer 기반 모델과 달리 선형적으로 증가함을 확인하였다 (Figure 4).

## 🧠 Insights & Discussion

### 강점 및 분석

- **선형 복잡도의 실용성**: Mamba의 선형 스케일링 특성 덕분에 메모리 제한 내에서 훨씬 더 많은 수의 프레임을 처리할 수 있다.
- **시공간 통합의 이점**: 프레임별로 독립적인 압축을 수행하는 것보다, TSTAR처럼 전체 시퀀스로 통합하여 압축하는 것이 토큰 효율성 면에서 압도적으로 유리하다.
- **외삽 능력 (Extrapolation)**: 학습 시에는 최대 128프레임을 사용했으나, 추론 시 256프레임 이상으로 확장해도 성능 저하가 완만하게 나타나, 긴 비디오에 대한 강건함을 보였다.

### 한계 및 논의

- **백본 의존성**: Qwen-2.5와 같은 더 강력한 LLM 백본을 사용할 때 성능이 일관되게 향상되는 것을 확인하였다. 이는 압축 모듈 자체의 성능뿐만 아니라 LLM의 기본 추론 능력이 최종 결과에 큰 영향을 미침을 시사한다.
- **데이터 밸런스**: Unified Training 시 이미지 데이터를 늘리는 것이 오히려 비디오 성능을 일부 저하시키는 현상이 발견되었으며, 이는 모달리티 간의 균형과 지시어 신호의 복잡성에 대한 추가 연구가 필요함을 의미한다.

## 📌 TL;DR

본 논문은 LMM에서 긴 비디오 처리 시 발생하는 토큰 폭발 문제를 해결하기 위해, SSM 기반의 **MambaMia** 압축 레이어와 계층적 샘플링 구조인 **TSTAR** 프레임워크를 제안한다. 이 방법은 기존 Transformer 기반 방식보다 훨씬 적은 토큰(예: 256프레임당 860개)을 사용하면서도 GPT-4V에 근접하는 성능을 내며, 특히 추론 비용이 선형적으로 증가하여 실제 환경에서의 배포 가능성을 크게 높였다.
