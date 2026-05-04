# Sparse Deformable Mamba for Hyperspectral Image Classification

Lincoln Linlin Xu, Yimin Zhu, Zack Dewis, Zhengsen Xu, Motasem Alkayid, Mabel Heffring, Saeid Taleghanidoozan (2025)

## 🧩 Problem to Solve

하이퍼스펙트럴 이미지(Hyperspectral Image, HSI) 분류는 고차원 데이터, 노이즈, 공간-분광적 이질성(spatial-spectral heterogeneity), 그리고 제한된 학습 샘플 수로 인해 매우 까다로운 과제이다. 최근 Mamba 모델이 HSI 분류 성능을 크게 향상시켰으나, Mamba의 핵심인 토큰 시퀀스(token sequence)를 효율적으로 구축하는 데 어려움이 있다.

기존의 Mamba 모델들은 대부분 미리 정의된(predefined) 결정론적 방식으로 토큰을 스캔한다. 이러한 dense한 스캔 방식은 데이터의 중복성을 초래하고, 모델의 유연성을 떨어뜨리며, 불필요한 계산 비용을 증가시킨다. 따라서 본 논문의 목표는 토큰 시퀀스를 적응적(adaptive)이고 희소하게(sparse) 구성하여, 계산 비용을 줄이면서도 HSI의 세부 특징 보존 능력을 높이는 Sparse Deformable Mamba (SDMamba) 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Sparse Deformable Sequencing (SDS)**이다. 이는 모든 토큰을 순차적으로 사용하는 대신, 데이터로부터 학습 가능한 방식으로 가장 관련성이 높은 소수의 토큰만을 선택하여 시퀀스를 구성하는 것이다.

주요 기여 사항은 다음과 같다:
1. **SDS 설계**: 결정론적인 dense 스캔 대신, 관련 토큰만을 적응적으로 학습하여 선택하는 SDS 방식을 통해 중복성을 제거하고 계산 효율성을 높였다.
2. **전용 모듈 구축**: SDS를 기반으로 공간 정보를 모델링하는 Sparse Deformable Spatial Mamba Module (SDSpaM)과 분광 정보를 모델링하는 Sparse Deformable Spectral Mamba Module (SDSpeM)을 각각 설계하였다.
3. **Attention 기반 특징 융합**: SDSpaM과 SDSpeM의 출력을 효율적으로 통합하기 위해 attention 메커니즘 기반의 특징 융합(feature fusion) 방식을 도입하였다.

## 📎 Related Works

HSI 특징 추출을 위해 과거에는 PCA(주성분 분석)나 ICA(독립 성분 분석)와 같은 특징 공학(feature engineering) 방식이 사용되었으나, 이는 데이터의 변별력 있는 정보를 적응적으로 캡처하는 데 한계가 있었다. 이후 CNN 기반 모델들이 등장하여 공간-분광 특징 학습을 개선했으나, 강한 귀납적 편향(inductive bias)과 지역성(locality)으로 인해 장거리 상관관계(long-range correlation)를 포착하는 능력이 부족했다.

Transformer는 장거리 문맥 파악에 유연하지만, attention matrix의 크기로 인해 계산 비용이 매우 높다는 단점이 있다. Mamba 모델은 토큰 시퀀싱을 통해 계산량을 획기적으로 줄이면서도 장거리 모델링 능력을 유지할 수 있어 대안으로 제시되었으나, 본 논문에서 지적하듯 시퀀스 구성 방식의 경직성과 중복성 문제가 여전히 존재한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
SDMamba는 입력 HSI 데이터 큐브 $X_j$를 받아 Stem layer를 거친 후, SDSpaM과 SDSpeM 모듈을 병렬적으로 통과시킨다. 이후 Attention Data Fusion Module을 통해 두 모듈의 특징을 융합하여 최종적으로 중심 픽셀의 클래스 레이블 $L_j$를 예측한다.

### 2. Sparse Deformable Sequencing (SDS)
SDS는 모든 토큰을 사용하는 대신, 중요도가 높은 토큰만 선택하여 Mamba Block에 입력한다. 이를 위해 adaptive attention matrix를 사용하여 토큰을 정렬하고 제한된 수의 토큰만 추출함으로써 sparsity와 deformability를 동시에 달성한다.

### 3. Sparse Deformable Spatial Mamba Module (SDSpaM)
공간 정보를 학습하기 위한 모듈로, 다음과 같은 절차를 따른다.
- **Stem Layer**: 입력 $X_j$를 통해 초기 특징 맵 $Y_j$를 생성한다.
$$Y_j = \text{GELU}(\text{BatchNorm}(\text{Conv}(X_j)))$$
- **Sparsity 구현**: $Y_j$를 $HW \times C$ 행렬 $Z_j$로 재구성하고, 중심 토큰 $z_j$를 앵커(anchor)로 설정하여 모든 토큰과의 코사인 유사도를 측정한다.
$$\text{SparseSpatialAttn}_i = \arccos \left( \frac{z_j^\top z_i}{\|z_j\| \|z_i\|} \right)$$
- **토큰 선택**: 위 식을 통해 계산된 유사도 값을 기준으로 정렬한 후, 희소성 비율 $\lambda$ (본 논문에서는 0.3)를 적용하여 상위 토큰(예: 5개)만 선택해 Mamba Block에 입력한다. 이후 결과값은 residual skip connection을 통해 원래 공간 차원으로 다시 분산(scatter)된다.

### 4. Sparse Deformable Spectral Mamba Module (SDSpeM)
분광 정보를 학습하기 위한 모듈이며, SDSpaM과 유사한 메커니즘을 사용하지만 대상이 분광 차원이다.
- **Sparsity 구현**: $Y_j$를 $C \times HW$ 행렬 $A_j$로 재구성하고, 무작위 토큰 $a_j$를 앵커로 설정하여 코사인 유사도를 측정한다.
$$\text{SparseSpectralAttn}_i = \arccos \left( \frac{a_j^\top a_i}{\|a_j\| \|a_i\|} \right)$$
- **토큰 선택**: 마찬가지로 $\lambda=0.3$을 적용하여 소수의 관련 토큰(예: 3개)만을 선택해 Mamba Block에 입력하고, 이를 residual connection으로 연결한다.

### 5. Attention Data Fusion Module
SDSpaM과 SDSpeM의 출력(각 $H \times W \times 256$)을 융합하기 위해 attention 메커니즘을 사용한다.
- SDSpaM의 출력으로 Query ($Q$)를 생성하고, SDSpeM의 출력으로 Key ($K$)와 Value ($V$)를 생성한다.
- $Q$와 $K$의 곱을 통해 attention matrix를 구하고, 이를 $V$에 적용하여 최종 융합된 특징 맵 $Y_j$를 생성한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Indian Pines (IP), Pavia University (PU)
- **평가 지표**: 전체 정확도 (OA), 평균 정확도 (AA), Kappa 계수
- **비교 대상**: SSRN, SS-ConvNeXt, MTGAN, SSFTT, SSTN, GSC-ViT, MambaHSI, HyperMamba, 3DSS-Mamba 등 최신 SOTA 모델들.

### 2. 정량적 결과
- **정확도**: SDMamba는 IP와 PU 데이터셋 모두에서 OA, AA, Kappa 모든 지표에서 타 모델들을 압도하는 성능을 보였다. 특히 **AA(평균 정확도)**에서 큰 향상을 보였는데, 이는 본 모델이 샘플 수가 적은 소수 클래스(small classes)를 보존하고 분류하는 능력이 뛰어남을 의미한다.
- **계산 효율성**: Ablation study 결과, sparsity ratio를 적용했을 때 dense한 방식보다 FLOPs가 현저히 낮음에도 불구하고 더 높은 정확도를 기록했다. 예를 들어, IP 데이터셋에서는 토큰의 5%만 사용해도 기존의 dense Mamba 방식보다 성능이 좋았다.

### 3. 정성적 결과
- **분류 맵**: 시각화 결과, SDMamba는 Ground Truth와 가장 일치하는 맵을 생성하였으며, 특히 클래스 간의 경계선(boundaries)과 소수 클래스의 영역을 더 정밀하게 묘사하였다.
- **t-SNE 시각화**: 추출된 특징들을 t-SNE로 시각화한 결과, 서로 다른 클래스들이 원래 공간보다 더 명확하게 분리(disentangle)되어 분포함을 확인하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 Mamba 모델의 토큰 시퀀싱 과정을 '고정된 스캔'에서 '학습 가능한 희소 선택'으로 전환했다는 점이다. 이를 통해 다음과 같은 통찰을 얻을 수 있다.

첫째, HSI 데이터의 모든 토큰이 분류에 유용한 것은 아니며, 오히려 불필요한 토큰(redundancy)이 모델의 성능을 저하시킬 수 있다. SDS를 통해 핵심 토큰만 선택함으로써 노이즈를 줄이고 유의미한 특징에 집중하게 만들었다.

둘째, 고정된 스캔 방식은 데이터의 기하학적 특성을 반영하지 못하는 경직성(rigidity)을 가지지만, 본 모델의 deformable sequencing은 데이터의 특성에 따라 유연하게 시퀀스를 변경함으로써 모델링 능력을 극대화했다.

셋째, 계산 복잡도를 획기적으로 줄이면서도 정확도를 높였다는 점은 실시간 HSI 분석 시스템으로의 확장 가능성을 시사한다. 다만, 앵커 토큰을 선택하는 방식(중심 픽셀 또는 무작위 픽셀)이 최종 성능에 어느 정도 영향을 미치는지에 대한 세부 분석은 부족한 것으로 보인다.

## 📌 TL;DR

본 논문은 Mamba 모델의 경직된 토큰 시퀀싱 문제를 해결하기 위해, 관련성 높은 토큰만 적응적으로 선택하는 **Sparse Deformable Sequencing (SDS)** 기법을 제안하였다. 이를 공간(SDSpaM) 및 분광(SDSpeM) 모듈에 적용하고 Attention 기반으로 융합한 **SDMamba**는 기존 SOTA 모델 대비 계산 비용을 크게 낮추면서도, 특히 소수 클래스 분류 성능을 획기적으로 향상시켰다. 이 연구는 효율적인 시퀀스 모델링이 고차원 데이터 분석에서 핵심적인 역할을 할 수 있음을 입증하였다.