# Spectral-Spatial Mamba for Hyperspectral Image Classification

Lingbo Huang, Yushi Chen, and Xin He (2021/2024)

## 🧩 Problem to Solve

본 논문은 초분광 이미지(Hyperspectral Image, HSI) 분류에서 발생하는 계산 복잡도와 특징 추출의 효율성 문제를 해결하고자 한다. HSI는 수백 개의 좁은 spectral band를 통해 풍부한 공간적(spatial) 및 분광적(spectral) 정보를 제공하지만, 데이터의 차원이 매우 높아 처리가 까다롭다.

최근 Transformer 기반 모델들이 HSI의 spatial-spectral 특징 간의 long-range dependencies를 모델링하는 데 탁월한 성능을 보였으나, self-attention 메커니즘으로 인한 이차 복잡도(quadratic computational complexity) 문제가 발생한다. 이는 고차원 HSI 데이터를 처리할 때 연산 비용을 급격히 증가시켜 실용성을 저해한다. 따라서 본 연구의 목표는 Transformer 수준의 모델링 능력을 유지하면서도 계산 효율성이 뛰어난 State Space Model(SSM) 기반의 Mamba 구조를 HSI 분류에 도입하여, 효율적이고 정확한 spectral-spatial 특징 추출 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 선형 시간 복잡도(linear-time complexity)를 활용하여 HSI의 고차원 시퀀스를 효율적으로 처리하는 **Spectral-Spatial Mamba (SS-Mamba)** 프레임워크를 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **SS-Mamba 프레임워크 제안**: Mamba의 효율적인 연산 능력과 강력한 long-range feature extraction 능력을 HSI 분류에 적용하였다.
2.  **Spectral-Spatial Token Generation 메커니즘**: HSI 큐브를 공간 토큰(spatial tokens)과 분광 토큰(spectral tokens) 시퀀스로 변환하는 토큰 생성 방식을 설계하여, spectral-spatial 정보를 최대한 활용할 수 있도록 하였다.
3.  **Feature Enhancement Module 설계**: HSI 샘플의 중심 영역(center region) 정보를 이용하여 공간 및 분광 토큰을 변조(modulation)함으로써, 정보가 집중된 영역에 주목하고 블록 내에서 효과적인 정보 융합을 달성하도록 하였다.

## 📎 Related Works

기존의 HSI 분류 연구는 크게 세 단계로 발전해 왔다. 초기에는 PCA, LDA와 같은 선형/비선형 변환을 통한 spectral feature 추출에 집중했으나, 공간 정보를 무시한다는 한계가 있었다. 이를 극복하기 위해 EMP, Gabor filtering 등 spectral-spatial 특징 추출 기법과 SVM 등의 분류기가 결합된 방식이 등장했으나, 수작업으로 설계된 특징(manually crafted features)에 의존하여 일반화 능력이 떨어졌다.

이후 딥러닝의 발전으로 CNN 기반 모델(1D, 2D, 3D CNN)이 자동화된 특징 추출을 통해 높은 성능을 보였다. 최근에는 long-range dependencies를 포착하기 위해 Transformer 기반 모델들이 도입되었으나, 앞서 언급한 것처럼 self-attention의 높은 계산 복잡도가 가장 큰 병목 현상으로 지적되었다. 본 논문은 이러한 Transformer의 한계를 극복하기 위해, 계산 효율성이 뛰어나면서도 유사한 모델링 능력을 가진 State Space Model(SSM) 기반의 Mamba를 대안으로 제시하며 기존 접근 방식과 차별화한다.

## 🛠️ Methodology

### 1. State Space Models (SSM) 개요
SS-Mamba의 근간이 되는 SSM은 입력 시퀀스 $x(t)$를 상태 변수 $h(t)$를 통해 출력 시퀀스 $y(t)$로 매핑하는 프레임워크이다. 이는 다음과 같은 상미분 방정식(ODE)으로 정의된다.

$$\dot{h}(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

여기서 $A \in \mathbb{R}^{N \times N}$는 상태 행렬, $B \in \mathbb{R}^{N \times 1}$와 $C \in \mathbb{R}^{N \times 1}$는 시스템 파라미터이다. 이를 이산화(discretization)하기 위해 Zero Order Holding (ZOH) 규칙을 적용하여 이산 파라미터 $\bar{A}$와 $\bar{B}$를 도출하며, 최종적으로 순환(recurrence) 형태나 합성곱(convolution) 형태로 계산하여 병렬 학습을 가능하게 한다. Mamba는 여기서 선택적 스캔(selective scan) 메커니즘을 추가하여 입력 데이터에 따라 동적으로 문맥 관계를 모델링한다.

### 2. Spectral-Spatial Tokens Generation
Mamba의 시퀀스 모델링 능력을 활용하기 위해 HSI 큐브 $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$를 두 가지 토큰 시퀀스로 변환한다.

*   **Spatial Token Generation**:
    1.  **Spectral Mapping**: HSI 샘플을 $HW \times B$로 변환 후 경량 MLP를 통해 $HW \times D'$ 차원으로 매핑하여 분광 정보를 먼저 처리한다.
    2.  **Spatial Partition**: 매핑된 텐서를 중첩되지 않는 패치(patch) $\bar{P}_{spa}$로 분할한다.
    3.  **Patch Embedding**: 선형 레이어를 통해 각 패치를 고정 차원 $D$의 공간 토큰 $\mathcal{T}_{spa}^0$로 투영한다.
*   **Spectral Token Generation**:
    1.  **Center Region Extraction**: 중심 영역 $\hat{\mathcal{X}} \in \mathbb{R}^{s \times s \times B}$ (여기서 $s=3$)를 추출하여 강건함을 높인다.
    2.  **Spatial Mapping**: 이를 $B \times s^2$로 변환 후 MLP를 통해 $B \times D'$로 매핑하여 공간 정보를 먼저 처리한다.
    3.  **Spectral Partition & Embedding**: 분광 차원을 따라 패치로 분할하고 선형 레이어를 통해 분광 토큰 $\mathcal{T}_{spe}^0$를 생성한다.

최종적으로 공간 토큰에는 2D sinusoidal positional embedding을, 분광 토큰에는 1D positional embedding을 추가한다.

### 3. Spectral-Spatial Mamba Block
본 모델은 여러 개의 SS-Mamba 블록이 적층된 구조이다. 각 블록은 두 개의 기본 Mamba 블록과 하나의 Feature Enhancement Module로 구성된다.

1.  **Feature Extraction**: 공간 토큰과 분광 토큰이 각각 독립적인 Mamba 블록을 통과하여 특징을 추출한다.
    $$\mathcal{T}_{spe}^l = \text{Mamba}_{spe}^l(\mathcal{T}_{spe}^{l-1}), \quad \mathcal{T}_{spa}^l = \text{Mamba}_{spa}^l(\mathcal{T}_{spa}^{l-1})$$
2.  **Feature Enhancement Module**: 
    *   공간 토큰 중 중심 패치 토큰 $\mathcal{V}_1^l$과 분광 토큰들의 평균값 $\mathcal{V}_2^l$을 구하여 평균을 낸다: $\mathcal{V}_{fus} = (\mathcal{V}_1^l + \mathcal{V}_2^l) / 2$.
    *   이를 $\text{Sigmoid}(\text{MLP}(\mathcal{V}_{fus}))$에 통과시켜 변조 가중치 $\mathcal{W}_{fus}$를 생성한다.
    *   이 가중치를 각각의 토큰 시퀀스에 원소별 곱셈(element-wise multiplication)하여 중요 영역에 집중하게 한다.
    $$\mathcal{T}_{spa}^l = \mathcal{W}_{spa}^l \otimes \mathcal{T}_{spa}^l, \quad \mathcal{T}_{spe}^l = \mathcal{W}_{spe}^l \otimes \mathcal{T}_{spe}^l$$

### 4. 최종 분류
모든 블록을 통과한 후, 최종 공간 토큰의 평균 $\mathcal{V}_{spa}$와 분광 토큰의 평균 $\mathcal{V}_{spe}$를 더해 최종 특징 벡터 $\mathcal{V} = \mathcal{V}_{spa} + \mathcal{V}_{spe}$를 형성한다. 이를 Fully Connected (FC) 레이어에 입력하여 최종 클래스 확률을 예측하며, Cross-Entropy Loss를 통해 학습한다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: Indian Pines, Pavia University, Houston, Chikusei 등 4종의 널리 사용되는 HSI 데이터셋을 사용하였다.
*   **비교 대상**: EMP-SVM(전통적 방식), CNN, SSRN, DBDA(CNN 기반), MSSG(Graph CNN), LSFAT, SSFTT, CT-Mixer(Transformer 기반) 등 다양한 모델과 비교하였다.
*   **평가 지표**: Overall Accuracy (OA), Average Accuracy (AA), Kappa coefficient (K)를 사용하였다.

### 2. 정량적 결과
모든 데이터셋에서 SS-Mamba는 비교 대상 모델들보다 우수한 성능을 보였다.
*   **성능 우위**: 특히 Transformer 기반 모델(SSFTT, CT-Mixer 등)보다 높은 OA를 기록하였다. 예를 들어, Houston 데이터셋에서 SS-Mamba의 OA는 $94.30\%$로, 다른 Transformer 기반 모델들이 $93\%$ 미만인 것과 대조적이다.
*   **Ablation Study**: 
    *   **시퀀스 모델 비교**: Mamba 기반 모델이 LSTM, GRU, Transformer보다 더 높은 정확도를 보였다.
    *   **모듈 효과**: Feature Enhancement Module을 제거했을 때보다 적용했을 때 성능이 유의미하게 상승하였다 (Houston 데이터셋 기준 OA $2.09\%$p 상승).
    *   **학습 방향**: Spectral-only나 Spatial-only 모델보다 Spectral-Spatial 통합 모델의 성능이 압도적으로 높았다.

### 3. 정성적 결과 및 효율성
*   **Classification Maps**: 시각화 결과, SS-Mamba는 다른 모델들이 혼동하기 쉬운 인접 지역(예: Pavia 데이터셋의 Asphalt, Trees, Bricks)을 더 정확하게 구분해냈다.
*   **복잡도 분석**: Pavia University 데이터셋 테스트 결과, Mamba는 Transformer보다 추론 시간이 빠르며 파라미터 수도 매우 적은 수준($0.47\text{M}$)으로 효율적임을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석
본 논문은 Mamba의 선형 시간 복잡도와 long-range modeling 능력이 HSI의 고차원 데이터 처리에 매우 적합함을 보여주었다. 특히, 단순한 적층 구조가 아니라 공간-분광 두 경로를 분리하여 처리하고 이를 중심 영역 정보로 융합하는 구조가 HSI의 특성을 잘 반영한 설계라고 판단된다. 또한, 데이터 샘플이 제한적인 상황에서 Transformer보다 높은 성능을 보인 점은 Mamba가 HSI 분류에서 더 강건한 특징 추출기일 가능성을 시사한다.

### 2. 한계 및 비판적 논의
*   **토큰 생성의 단순성**: 저자들도 언급했듯이, 현재의 패치 분할 방식은 객체의 방향이나 모양을 고려하지 않는다. 이로 인해 동일 객체가 서로 다른 패치로 쪼개져 시맨틱 구조가 일부 파괴될 위험이 있다.
*   **모델 깊이**: Feature map 분석 결과, 모델의 깊이가 얕아 층이 깊어짐에 따라 특징이 점진적으로 추상화되는 양상이 뚜렷하지 않다. 더 깊은 구조에서의 성능 변화에 대한 분석이 추가될 필요가 있다.
*   **Spectral Unmixing과의 비교**: 본 모델은 end-to-end 방식으로 효율적이지만, 물리적 기반의 spectral unmixing 기법을 결합했을 때 성능이 얼마나 더 향상될 수 있을지에 대한 실증적 연구가 부족하다.

## 📌 TL;DR

본 논문은 Transformer의 높은 계산 비용 문제를 해결하기 위해 **State Space Model 기반의 Mamba를 도입한 HSI 분류 모델(SS-Mamba)**을 제안하였다. HSI 데이터를 공간 및 분광 토큰 시퀀스로 변환하고, 이를 Mamba 블록과 중심 영역 기반의 특징 강화 모듈로 처리함으로써 **계산 효율성과 분류 정확도를 동시에 확보**하였다. 실험 결과, 기존 CNN 및 Transformer 기반 모델들을 능가하는 성능을 보였으며, 이는 향후 고해상도/고차원 원격 탐사 데이터 처리 분야에서 Mamba 계열 모델의 활용 가능성을 제시한 중요한 연구이다.