# Low-bit Model Quantization for Deep Neural Networks: A Survey

Kai Liu, Qian Zheng, Kaiwen Tao, et al. (2025)

## 🧩 Problem to Solve

본 논문은 딥 뉴럴 네트워크(DNN)의 실용적인 배포를 가로막는 핵심 장애물인 막대한 계산 비용과 거대한 모델 크기 문제를 해결하고자 한다. DNN은 이미지 인식, 자연어 처리 등 다양한 분야에서 탁월한 성능을 보이지만, 높은 메모리 점유율과 연산량으로 인해 자원이 제한된 엣지 디바이스나 실시간 시스템에 적용하기 어렵다.

모델 양자화(Model Quantization)는 연속적인 부동 소수점(Floating-point) 숫자를 이산적인 정수(Integer) 형태로 변환하여 메모리 I/O 및 연산 속도를 획기적으로 높이는 효과적인 가중치 경량화 기술이다. 그러나 이러한 변환 과정에서 정밀도 손실(Loss of precision)이 발생하며, 이는 모델의 성능 저하로 이어진다. 따라서 본 논문의 목표는 최근 5년간의 저비트(Low-bit) 양자화 연구 동향을 종합적으로 분석하고, 정밀도 손실을 보상하며 효율적인 변환을 수행하는 최신 방법론들을 체계적으로 분류하여 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 최근 5년간 발표된 179편의 양자화 관련 논문을 심도 있게 분석하여, 이를 핵심 기술에 따라 **8개의 주요 카테고리와 24개의 세부 카테고리로 분류한 체계적인 텍스노미(Taxonomy)**를 구축한 것이다. 

단순한 방법론 나열에 그치지 않고, 양자화의 기본 수학적 원리부터 시작하여 최신 LLM(Large Language Models) 및 Diffusion Model에 적용되는 특수 양자화 기법까지 포괄적으로 다룬다. 특히 정밀도 유지와 모델 압축 사이의 Pareto frontier를 달성하기 위한 최신 전략들을 비교 분석함으로써, 향후 연구자들이 참고할 수 있는 포괄적인 가이드를 제공한다.

## 📎 Related Works

논문은 기존의 양자화 서베이 연구들과의 차별점을 다음과 같이 명시한다.

- **기존 연구의 한계**: Rokh et al.의 연구는 주로 이미지 분류 작업에 국한되어 있으며, Gholami et al.의 연구는 일반적인 추론 효율성에 집중했다. 또한 Shen et al.의 서베이는 LLM에 특화되어 있어 다른 최신 모델(예: Vision Transformer, Diffusion Model)에 대한 포괄성이 부족하다.
- **본 논문의 차별점**: 본 연구는 특정 태스크나 모델에 국한되지 않고, 최신 딥러닝 아키텍처 전반을 아우르는 최신 저비트 양자화 기법을 다룬다. 특히 최근 급격히 성장한 LLM과 Diffusion Model의 양자화 이슈를 포함하여 가장 최신의 연구 흐름을 반영하였다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 양자화의 수학적 정식화 (Formalization)

양자화는 부동 소수점 입력 $X^{FP}$를 양자화 연산자 $Q(\cdot)$를 통해 저정밀도 범위로 변환하는 과정이다. 이 과정에는 비트 너비 $b$, 스케일 팩터 $s$, 제로 포인트 $z$라는 세 가지 파라미터가 사용된다.

**양자화 과정(Quantization):**
$$X^{int} = \text{Clamp}(\text{Round}(\frac{X^{FP}}{s}) + z, n, p)$$
여기서 $\text{Clamp}(X, n, p)$는 값을 $[n, p]$ 범위 내로 제한하는 함수이며, $(n, p)$는 비트 너비 $b$에 의해 결정되는 정수 그리드 범위이다.

**역양자화 과정(De-quantization):**
$$X^{FP} \approx \hat{X} = s(X^{int} - z)$$

양자화의 목표는 저장 공간 제한 $\text{Storage}(\hat{M}) \le C$를 만족하면서 테스트 데이터셋에 대한 평가 손실 $L$을 최소화하는 모델 $\hat{M}$을 찾는 것이다. 실제 구현에서는 각 레이어별로 다음과 같은 MSE 손실을 최소화하는 방식을 주로 사용한다:
$$\min_Q \|Q(X)Q(W) - XW\|_2^2$$

### 2. 양자화 분류 체계 (Taxonomy)

논문은 양자화 방법론을 다음과 같은 기준으로 분류한다.

- **대칭성**: $q_{max} = -q_{min}$인 Symmetric Quantization과 제로 포인트 $z$를 허용하는 Asymmetric Quantization으로 나뉜다.
- **균등성**: 간격이 일정한 Uniform Quantization과 데이터 분포에 따라 간격을 조절하는 Non-Uniform Quantization으로 구분된다.
- **결정 시점**: 추론 시 실시간으로 계산하는 Dynamic Quantization과 사전 계산된 값을 사용하는 Static Quantization으로 나뉜다.
- **입도(Granularity)**: Per-tensor(레이어 단위) $\rightarrow$ Per-group $\rightarrow$ Per-channel $\rightarrow$ Per-token 순으로 정밀도가 높아진다.
- **학습 시점**: 사전 학습된 모델에 적용하는 PTQ(Post-Training Quantization)와 학습 과정에서 양자화를 고려하는 QAT(Quantization-Aware Training)로 구분된다.

### 3. 8대 핵심 고급 주제 (Advanced Topics)

1. **Better $s$ and $z$**: MSE나 EMA 같은 단순 지표 대신 task loss나 quantization loss를 최적화하여 최적의 스케일 팩터를 찾는 방법론들이다.
2. **Metric and Mechanism**: QAT 과정에서의 가중치 진동(Oscillation) 문제를 해결하거나 새로운 손실 함수를 도입하는 기법들이다.
3. **Mixed Precision**: 레이어의 중요도에 따라 서로 다른 비트 너비를 할당하여 효율성과 정확도의 균형을 맞춘다.
4. **Redistribution**: Outlier(이상치)의 영향을 줄이기 위해 Hadamard Transform 같은 가역 행렬을 사용하여 데이터 분포를 평탄화한다.
5. **Data-Free Quantization (DFQ)**: 개인정보 보호 등의 이유로 원본 데이터가 없을 때, 합성 데이터를 생성하거나 가중치 통계만을 이용하여 양자화한다.
6. **Advanced Format**: 표준 정수 형식을 넘어 NF(NormalFloat)나 Block Floating Point 같은 특수 포맷을 사용한다.
7. **Diffusion Model**: 타임스텝(Time-step)에 따라 변화하는 활성화 값 분포와 누적 오차 문제를 해결하는 특수 기법들을 다룬다.
8. **Other**: 정수 연산만으로 구성하는 Full-quantization, 타 압축 기술(Pruning 등)과의 결합 등을 포함한다.

## 📊 Results

본 논문은 서베이 논문이므로 특정 실험 결과보다는 **기존 연구들의 분석 결과**를 종합하여 제시한다.

- **분석 대상**: 최근 5년간의 양자화 관련 논문 179편을 분석하였다.
- **정성적 결과**: 
    - **LLM 양자화**: 활성화 값의 Outlier 채널 문제가 심각하며, 이를 해결하기 위해 SmoothQuant와 같은 Redistribution 기법이 필수적임을 확인하였다.
    - **Diffusion 모델 양자화**: 타임스텝별로 활성화 분포가 달라지므로, 이에 적응하는 Dynamic Quantization이나 타임스텝 인식(Time-step aware) 보정 기법이 성능 향상에 결정적임을 분석하였다.
    - **데이터 프리 양자화**: BN(Batch Normalization) 층의 통계 정보를 이용한 데이터 생성이 가능하지만, 생성된 데이터의 다양성을 확보하는 것이 성능의 핵심임을 밝히고 있다.

## 🧠 Insights & Discussion

### 강점 및 통찰
- **포괄적 체계화**: 파편화되어 있던 다양한 양자화 기법들을 8대 카테고리와 24개 세부 항목으로 체계화하여, 연구자들이 자신의 문제에 맞는 기법을 빠르게 찾을 수 있도록 돕는다.
- **최신 트렌드 반영**: 단순한 CNN/RNN 시대를 넘어 ViT, LLM, Diffusion Model이라는 최신 아키텍처의 특성에 따른 양자화 전략의 변화를 정확히 짚어내었다. 특히 Outlier 처리와 Redistribution의 중요성을 강조한 점이 돋보인다.

### 한계 및 비판적 해석
- **정량적 비교의 부재**: 논문에서도 언급되었듯이, 각 연구마다 사용한 모델, 데이터셋, 평가 지표가 너무 다양하여 직접적인 성능 비교 표를 제시하지 못한 점은 아쉽다. 이는 양자화 분야의 벤치마크 표준화가 필요함을 시사한다.
- **하드웨어 구현의 괴리**: 많은 방법론이 수학적으로는 우수하지만, 실제 GPU/NPU 하드웨어에서 정수 연산 가속기로 구현했을 때 동일한 속도 향상을 얻을 수 있는지에 대한 실증적 분석은 부족한 편이다.

## 📌 TL;DR

본 논문은 최신 딥러닝 모델의 경량화를 위한 **저비트 양자화 기술의 최신 동향을 정리한 종합 서베이 보고서**이다. 총 179편의 논문을 분석하여 **8대 카테고리 및 24개 세부 분류**라는 체계적인 텍스노미를 제시하였으며, 특히 LLM과 Diffusion 모델에서 발생하는 Outlier 문제와 타임스텝별 분포 변화를 해결하기 위한 최신 전략들을 상세히 다루고 있다. 이 연구는 향후 멀티모달 모델의 효율적인 배포와 소프트웨어-하드웨어 통합 최적화 연구에 중요한 기초 자료로 활용될 가능성이 매우 높다.