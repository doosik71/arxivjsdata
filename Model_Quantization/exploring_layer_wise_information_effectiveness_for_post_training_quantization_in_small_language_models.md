# Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small Language Models

He Xiao, Qingyao Yang, Dirui Xie, Wendong Xu, Zunhai Su, Runming Yang, Haobo Liu, Wenyong Zhou, Zhengwu Liu, Ngai Wong (2025)

## 🧩 Problem to Solve

본 논문은 소형 언어 모델(Small Language Models, SLMs, 특히 파라미터 8B 미만 모델)을 대상으로 한 극단적인 저비트 양자화(Ultra-low bit quantization) 시 발생하는 성능 저하 문제를 해결하고자 한다.

**1. 해결하고자 하는 문제**

- **정확도 붕괴(Accuracy Collapse):** 8B 미만의 SLM은 대형 모델에 비해 중복성(Redundancy)이 부족하여, 2비트 수준의 양자화를 적용할 경우 Perplexity(PPL)가 급격히 상승하는 등 심각한 성능 저하가 발생한다.
- **하드웨어 효율성 저하:** 기존의 혼합 정밀도(Mixed-precision) 양자화 방식들은 요소별(Element-wise) 또는 그룹별(Group-wise)로 비트를 다르게 할당하여 정확도를 높이려 하지만, 이는 가중치 레이아웃을 불규칙하게 만들어 텐서 연속성을 해치고, 표준 커널(Standard kernels) 사용을 어렵게 하며 추론 지연 시간을 증가시킨다.

**2. 문제의 중요성**
엣지 디바이스(스마트폰, 로봇, 드론 등)의 제한된 메모리 예산(보통 4~12GB) 내에서 LLM을 구동하기 위해서는 공격적인 압축이 필수적이다. 따라서 하드웨어 효율성을 유지하면서도 정확도 손실을 최소화하는 양자화 기법이 필요하다.

**3. 논문의 목표**
하드웨어 친화적인 규칙적인 가중치 레이아웃(Regular weight layout)을 유지하면서, 레이어별 중요도에 따라 정밀도를 다르게 할당하여 SLM의 2비트 수준 양자화 성능을 극대화하는 PTQ(Post-Training Quantization) 프레임워크인 **LieQ**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"레이어의 기능적 중요성(Functional Saliency)과 표현의 압축성(Representational Compactness) 사이에 강한 상관관계가 있다"**는 통찰이다.

- **기하학적 대리 지표(Geometric Proxy) 발견:** 특정 레이어를 제거했을 때 발생하는 성능 저하(PPL 상승)를 측정하는 대신, 가중치 행렬의 특이값(Singular value) 분포를 통해 해당 레이어가 얼마나 많은 정보를 집중적으로 담고 있는지(Compactness)를 측정하여 중요도를 판별할 수 있음을 보였다.
- **LieQ 프레임워크 제안:** 위에서 발견한 기하학적 지표를 기반으로, 중요도가 높은 레이어에는 4비트를, 낮은 레이어에는 2비트를 할당하는 자동 비트 할당 전략을 제안한다.
- **하드웨어 친화적 설계:** 레이어 내부에서는 동일한 비트 너비를 유지하고 레이어 간에만 정밀도를 다르게 하여, 표준 행렬 곱셈 커널을 그대로 사용할 수 있도록 설계하였다.

## 📎 Related Works

**1. 기존 PTQ 및 혼합 정밀도 방식**

- GPTQ, RPTQ 등 초기 방식은 대형 모델에서는 효과적이었으나 SLM에서는 성능이 급격히 떨어진다.
- AWQ, OmniQuant 등은 활성화 값이나 보정(Calibration)을 통해 정확도를 높이지만, 일부 방식은 비정형 포맷을 사용하여 하드웨어 효율성을 희생한다.

**2. 극단적 저비트 양자화(2-bit)**

- QuIP, AQLM 등은 회전(Rotation)이나 코드북(Codebook) 기반 방식을 통해 2비트에서도 높은 정확도를 달성한다. 하지만 이러한 방식들은 추론 시 추가적인 변환 과정이나 불규칙한 메모리 접근이 필요하여 실제 배포 시 복잡도를 증가시킨다.

**3. 차별점**
LieQ는 별도의 런타임 변환이나 코드북 없이, **레이어 단위의 정밀도 혼합(Layer-wise mixed precision)**만을 사용하여 표준 커널 효율성을 유지하면서도 2비트 근처의 예산에서 SOTA급 성능을 목표로 한다.

## 🛠️ Methodology

### 1. 중요도 진단 지표: Representational Compactness

논문은 레이어의 중요도를 판단하기 위해 '기능적 중요성($\Delta PPL$)'의 대리 지표로 '표현 압축성($\Delta r$)'을 정의한다.

**가. 기능적 중요성 (Functional Saliency)**
레이어 $\ell$을 제거했을 때의 Perplexity 변화량 $\Delta PPL_\ell$로 측정한다.
$$\Delta PPL_\ell = PPL_{\setminus \ell} - PPL_{base}$$
하지만 이는 모든 레이어에 대해 개별적으로 추론을 수행해야 하므로 비용이 매우 크다.

**나. 표현 압축성 (Representational Compactness)**
학습된 가중치 $W$가 정보를 얼마나 효율적으로 집중시키고 있는지를 특이값 분해(SVD)를 통해 측정한다.

1. 학습된 가중치로 투영된 표현 $Z = W^{(\ell)} h^{(\ell)}$와 무작위 초기화된 가중치로 투영된 $\tilde{Z} = \tilde{W}^{(\ell)} h^{(\ell)}$를 생성한다.
2. 각각의 특이값 $\{\sigma_k\}$와 $\{\tilde{\sigma}_k\}$를 구하고, 정규화된 에너지 $p_k$를 계산한다.
   $$p_k = \frac{\sigma_k^2}{\sum_{j=1}^K \sigma_j^2}$$
3. 섀넌 엔트로피(Shannon Entropy)의 지수 함수를 이용하여 압축성(Compactness)을 정의한다.
   $$Compact(Z) = \exp\left( -\sum_{k=1}^K p_k \log p_k \right)$$
4. 최종적으로 학습 전후의 상대적 변화량 $\Delta r$을 레이어의 중요도 점수로 사용한다.
   $$\Delta r_\ell^{(P)} = \frac{Compact(\tilde{Z}) - Compact(Z)}{Compact(\tilde{Z})}$$
   $\Delta r$ 값이 클수록 학습을 통해 정보가 특정 매니폴드로 강하게 집중되었음을 의미하며, 이는 양자화 노이즈에 더 민감한(중요한) 레이어임을 시사한다.

### 2. LieQ의 비트 할당 및 추론 절차

**가. 레이어 선택**
각 레이어 $\ell$의 모든 선형 투영(Linear Projections)에 대해 $\Delta r$의 평균을 내어 최종 점수 $s_\ell$를 산출하고, 이를 내림차순으로 정렬하여 상위 $K$개의 레이어를 고정밀도 세트 $S_{hi}$로, 나머지를 저정밀도 세트 $S_{lo}$로 구분한다.

**나. 비트 할당 규칙**

- $\ell \in S_{hi} \implies b_\ell = 4\text{-bit}$
- $\ell \in S_{lo} \implies b_\ell = 2\text{-bit}$

**다. 메모리 예산 제어**
목표 평균 비트 예산 $\bar{b} \in [2, 4]$가 주어지면, 4비트로 할당할 레이어의 비율 $f = \frac{\bar{b}-2}{4-2}$를 계산하여 예산에 맞게 $S_{hi}$의 크기를 결정한다.

**라. 통합 및 추론**
LieQ는 특정 양자화 백엔드(예: GPTQ, AWQ)와 결합하여 사용할 수 있는 플러그앤플레이(Plug-and-play) 가이드라인 역할을 한다. 추론 시에는 각 레이어 내부의 비트 너비가 균일하므로 표준 행렬 곱셈 커널을 사용하여 빠르게 연산한다.

## 📊 Results

### 1. 실험 설정

- **대상 모델:** Qwen3 (0.6B, 1.7B, 4B, 8B), LLaMA 3.x (1B, 3B, 8B)
- **비교 대상:** GPTQ, AWQ, OmniQuant, PB-LLM, SliM-LLM, QuIP#, AQLM
- **평가 지표:** WikiText-2 및 C4 데이터셋의 Perplexity(PPL), 7가지 제로샷 추론 태스크(MMLU, ARC, PIQA 등)의 정확도.

### 2. 주요 결과

- **정확도 회복:** 2비트 수준의 예산에서 LieQ는 나이브한 2비트 baseline들이 보이는 심각한 성능 붕괴를 효과적으로 억제하였다. 특히 Qwen3와 LLaMA3 시리즈에서 FP16에 근접하거나 타 2-bit 방법론보다 월등한 성능을 보였다.
- **제로샷 추론 성능:** Table 3에 따르면, LLaMA-3-3B 모델에서 LieQ(2.07-bit)는 PIQA, ARC 등 다수 지표에서 FP16에 근접한 성능을 내며 타 방법론을 압도하였다.
- **하드웨어 효율성:** Microbenchmark 결과, LieQ는 FP16 대비 지연 시간(Latency)을 크게 줄였으며, 불규칙한 포맷을 사용하는 AQLM 등과 달리 표준 커널을 사용하여 처리량을 극대화하였다.
- **비트 예산 민감도:** 단 하나의 레이어만 4비트로 보호하더라도(Top-1 $\Delta r$), 단순 2비트 양자화보다 훨씬 높은 정확도 회복이 가능함을 확인하였다.

## 🧠 Insights & Discussion

**1. 강점 및 통찰**

- 본 연구는 양자화 민감도가 단순히 무작위적인 현상이 아니라, 가중치 매니폴드의 구조적 특성(Representational Geometry)에 뿌리를 두고 있음을 증명하였다.
- 비용이 많이 드는 PPL 프로빙(Perplexity probing) 없이, 단 한 번의 순전파(Forward pass)와 SVD만으로 중요 레이어를 식별할 수 있는 효율적인 지표를 제시하였다.

**2. 한계 및 논의사항**

- **엔지니어링 최적화:** 본 논문은 분석 프록시(Proxy)를 정립하는 데 집중하였기에, 실제 시스템 레벨에서의 엔드-투-엔드 처리량(Throughput) 최적화에는 여전히 개선의 여지가 있다.
- **가정:** 본 방법론은 훈련된 모델의 가중치 구조가 특정 정보를 압축하여 저장한다는 가설에 기반하며, 이는 대부분의 트랜스포머 모델에서 유효하지만 모델 아키텍처가 완전히 바뀔 경우 재검증이 필요할 수 있다.

## 📌 TL;DR

LieQ는 SLM의 극단적 저비트 양자화 시 발생하는 성능 저하를 막기 위해, **가중치의 기하학적 특성(표현 압축성)을 이용하여 중요 레이어를 식별하고 정밀도를 차등 할당(2-bit vs 4-bit)**하는 프레임워크이다. 이를 통해 하드웨어 효율성(표준 커널 사용)을 유지하면서도, 기존 2비트 양자화 방식 대비 획기적으로 향상된 정확도를 달성하였다. 이 연구는 향후 자원 제한적인 엣지 디바이스에 고성능 SLM을 배포하는 데 있어 실질적인 경로를 제공한다.
