# Mamba-in-Mamba: Centralized Mamba-Cross-Scan in Tokenized Mamba Model for Hyperspectral Image Classification

Weilian Zhou, Sei-ichiro Kamata, Haipeng and others (2024)

## 🧩 Problem to Solve

본 논문은 초분광 이미지(Hyperspectral Image, HSI) 분류에서 발생하는 기존 순차적 모델(Sequential Models)의 한계를 해결하고자 한다. HSI 분류는 각 픽셀에 지표면의 토지 피복 라벨을 할당하는 작업으로, 일반적으로 중심 픽셀 주변의 국소 패치(Patch)를 분석하는 'patch-wise' 학습 프레임워크를 사용한다.

기존의 접근 방식인 RNN(Recurrent Neural Networks)은 입력 시퀀스의 마지막 단계에 특징이 편향되는 성질이 있어 중심 픽셀의 특징을 효과적으로 집계하지 못하며, 주변의 간섭 픽셀에 민감하게 반응하는 문제가 있다. 반면, Transformer 기반 모델은 자기 주의 집중(Self-attention) 메커니즘으로 인해 계산 복잡도가 패치 크기의 제곱에 비례하여 증가하며, 특히 HSI 데이터셋처럼 학습 샘플이 제한적인 환경에서는 과적합(Overfitting) 위험이 크고 공간적 문맥을 포착하는 귀납적 편향(Inductive Bias)이 부족하다는 단점이 있다.

따라서 본 연구의 목표는 RNN의 효율성과 Transformer의 전역적 문맥 파악 능력을 동시에 갖춘 State Space Model(SSM), 특히 Mamba 아키텍처를 HSI 분류에 최적화하여 적은 자원으로도 높은 분류 정확도를 달성하는 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba 아키텍처를 HSI 패치 기반 분류 작업에 적합하도록 재설계한 **Mamba-in-Mamba (MiM)** 구조를 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Centralized Mamba-Cross-Scan (MCS)**: 이미지를 단순히 일렬로 나열하는 Raster scan 대신, 패치의 양 끝단에서 시작하여 중심 픽셀로 수렴하는 4가지 유형의 중심 집중형 스캔 방식을 도입하여 중심 픽셀의 특징을 효과적으로 보존한다.
2.  **Tokenized Mamba (T-Mamba) Encoder**: Mamba 인코더 내부에 Gaussian Decay Mask (GDM), Semantic Token Learner (STL), Semantic Token Fuser (STF)를 통합하여, 다중 스케일의 특징을 효과적으로 추출하고 정제한다.
3.  **Weighted MCS Fusion (WMF) 및 Multi-Scale Loss**: 서로 다른 스캔 방향에서 추출된 특징들을 동적으로 가중 결합하고, 하향 샘플링(Down-sampling) 과정을 통해 얻은 다중 스케일 특징들에 대해 개별적인 손실 함수를 적용하여 학습 효율을 극대화한다.

## 📎 Related Works

논문에서는 SSM(State Space Models)과 그 발전 형태인 Mamba에 대해 설명한다. S4(Structured State Space Sequence) 모델은 선형 시간 복잡도로 긴 시퀀스를 처리할 수 있지만, 입력 내용에 반응하지 않는 LTI(Linear Time Invariance) 특성을 가진다. 이를 개선한 Mamba(S6)는 선택적 스캔(Selective Scan) 메커니즘을 통해 입력 데이터에 따라 파라미터가 동적으로 변하게 하여 더 유연한 정보 처리를 가능하게 한다.

시각적 작업에 Mamba를 적용한 Vision Mamba (ViM)나 Visual Mamba (VMamba) 같은 연구들이 있었으나, 이를 HSI 분류에 직접 적용하기에는 한계가 있다. HSI 패치 기반 분류는 중심 픽셀의 예측이 핵심인데, 기존 모델들은 다중 스케일의 지각 능력이나 토지 피복의 시맨틱 토큰을 효과적으로 처리하는 메커니즘이 부족하여 HSI 데이터의 특성을 충분히 반영하지 못한다.

## 🛠️ Methodology

### 1. 전체 파이프라인
입력 HSI 패치는 먼저 PCA(주성분 분석), Depth-wise Convolution, Point-wise Convolution을 통해 차원 축소 및 지역 특징 추출 과정을 거친다. 이후 MCS를 통해 4가지 유형의 시퀀스로 변환되며, 각 시퀀스는 T-Mamba 인코더를 통과한다. 최종적으로 WMF 모듈을 통해 특징이 융합되며, 이 과정이 패치 크기가 1이 될 때까지 반복적으로 수행되는 계층적 구조를 가진다.

### 2. Centralized Mamba-Cross-Scan (MCS)
MCS는 패치의 공간적 연속성을 유지하기 위해 Snake scan 방식을 사용하며, 중심 픽셀 $x^{(i,j)}$를 향해 수렴하는 4가지 경로를 생성한다.
- **Type-1**: 좌상단 $\to$ 중심 / 우하단 $\to$ 중심 (Snake-like)
- **Type-2**: 좌상단 $\to$ 중심 / 우하단 $\to$ 중심 (Vertical-like)
- **Type-3**: 우상단 $\to$ 중심 / 좌하단 $\to$ 중심 (Horizontal-like)
- **Type-4**: 우상단 $\to$ 중심 / 좌하단 $\to$ 중심 (Vertical-like)

각 타입은 두 개의 하위 시퀀스(Forward, Backward)로 나뉘어 Mamba 모델에 입력되며, 중심 픽셀이 시퀀스의 마지막 단계에 위치하게 하여 이전 모든 단계의 특징이 중심 픽셀로 집계되도록 설계되었다.

### 3. Tokenized Mamba (T-Mamba) Encoder
T-Mamba는 단순한 인코딩을 넘어 특징을 정제하고 압축하는 과정을 포함한다.

- **Gaussian Decay Mask (GDM)**: 시퀀스 내의 각 단계에 가중치를 부여하는 소프트 마스크이다. 인덱스 거리 기반 가중치 $W_{idx}$와 특징 유사도(Euclidean distance) 기반 가중치 $W_{fea}$를 곱하여 중심 픽셀에 가까운 특징이 더 강조되도록 한다.
  $$w_t = \exp \left( -\frac{1}{2} \left( \frac{t-T}{\sigma_{idx}} \right)^2 \right)$$
- **Semantic Token Learner (STL)**: 시퀀스 특징 $\bar{s}$를 입력받아 대표 시맨틱 토큰 $u$를 생성한다. 1D Convolution과 Sigmoid 함수를 이용한 Sequential Attention을 적용해 중요한 특징을 강조한 후, 가우시안 분포로 초기화된 가중치 행렬 $U_1, U_2$를 이용해 특징을 압축한다.
- **Semantic Token Fuser (STF)**: 학습된 추상적 토큰 $u$를 원래의 시퀀스 특징 $z$와 융합한다. 영향력 점수 $\bar{z}$를 계산하여 토큰 정보를 시퀀스 차원으로 다시 확산시키고, attention-weighted 특징과 합산하여 최종 출력 $\bar{u}$를 생성한다.

### 4. Weighted MCS Fusion (WMF) 및 Multi-Scale Loss
4가지 MCS 경로에서 나온 결과 $o_1, o_2, o_3, o_4$를 학습 가능한 가중치 $k_n$을 사용하여 결합한다.
$$O^{(i,j)}_p = k_1 o^{(i,j)}_1 + k_2 o^{(i,j)}_2 + k_3 o^{(i,j)}_3 + k_4 o^{(i,j)}_4$$
또한, 모델은 패치 크기가 $p \to p_2 \to \dots \to 1$로 줄어드는 과정을 거치며, 각 스케일 단계마다 예측값 $Y^p$를 도출하고 이를 개별적인 Cross-Entropy Loss로 계산하여 전체 손실 함수를 구성한다.
$$L_{total} = \frac{1}{n} (L_p + L_{p_2} + \dots + L_{p_n})$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Indian Pines (IP), Pavia University (PU), Houston 2013 (HU2013), WHU-Hi-HongHu (HongHu).
- **비교 모델**: vanilla ViT, SpeFormer, MAEST, HiT, SSFTT, 3DViT, HUSST, ViM.
- **평가 지표**: 전체 정확도(OA), 평균 정확도(AA), Kappa 계수.

### 2. 주요 결과
- **정량적 성능**: MiM 모델은 4개의 모든 데이터셋에서 기존 베이스라인 및 SOTA 모델들과 경쟁하거나 이를 능가하는 성능을 보였다. 특히 IP 데이터셋에서 OA 92.08%, PU에서 91.58%를 기록하며 매우 높은 정확도를 달성하였다.
- **시각적 분석**: 분류 맵(Classification Map) 분석 결과, vanilla ViT에서 나타나는 노이즈 같은 아티팩트가 거의 없으며, land-cover 경계선이 매우 뚜렷하고 영역 내부가 매끄럽게 분류됨을 확인하였다.
- **계산 효율성**: 파라미터 수 측면에서 HiT, SSFTT, HUSST와 같은 복잡한 모델들보다 훨씬 효율적이며, 추론 시간과 모델 크기 사이의 최적의 균형을 이루었다.
- **t-SNE 시각화**: 학습된 특징 공간에서 클래스 간의 구분 능력이 매우 뛰어나며, 동일 클래스의 클러스터가 더 밀집되고 명확하게 형성됨을 보여주었다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
본 논문은 Mamba의 선형 시간 복잡도라는 이점을 유지하면서 HSI의 공간적 특성을 반영하기 위해 '중심 집중형 스캔'과 '시맨틱 토큰화'라는 전략을 성공적으로 결합하였다. 특히 제한된 학습 샘플 환경에서 Transformer보다 우수한 성능을 보인 점은 Mamba 계열 모델이 가진 귀납적 편향이 소규모 데이터셋의 HSI 분류 작업에 유리함을 시사한다.

### 2. 한계 및 비판적 해석
데이터 증강(Data Augmentation) 실험 결과, 학습 샘플 수가 6배 이상 증가한 환경에서는 MAEST와 같은 Transformer 기반 모델들이 MiM보다 더 높은 성능을 기록하였다. 이는 Transformer의 글로벌 어텐션 메커니즘이 충분한 데이터가 확보되었을 때 일반화 능력이 더 강력하다는 기존의 이론과 일치한다. 따라서 MiM은 **'데이터가 부족한 실제 현장 상황'**에서는 매우 강력하지만, 대규모 데이터셋이 준비된 상황에서는 Transformer 기반 모델에 밀릴 가능성이 있다.

### 3. 향후 연구 방향
저자들은 현재의 2D 스캔 방식을 확장하여, 분광 정보의 연속성을 함께 고려할 수 있는 3D Mamba 구성으로 발전시킬 계획임을 명시하고 있다.

## 📌 TL;DR

본 논문은 HSI 분류를 위해 Mamba 아키텍처를 기반으로 한 **Mamba-in-Mamba (MiM)** 모델을 제안한다. 중심 픽셀로 수렴하는 **Centralized Mamba-Cross-Scan**과 특징을 정제하는 **T-Mamba 인코더**, 그리고 **다중 스케일 손실 설계**를 통해 효율성과 정확도를 동시에 잡았다. 특히 학습 데이터가 제한적인 상황에서 SOTA급 성능을 보이며, 계산 복잡도를 낮추면서도 고해상도의 분류 맵을 생성할 수 있음을 입증하였다. 이는 향후 자원 제한적인 환경에서의 실시간 원격 탐사 이미지 분석에 중요한 역할을 할 것으로 기대된다.