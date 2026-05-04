# Directional Connectivity-based Segmentation of Medical Images

Ziyun Yang, Sina Farsiu (2023)

## 🧩 Problem to Solve

의료 영상 분석에서 바이오마커 분할(biomarker segmentation) 시 해부학적 일관성(anatomical consistency)을 유지하는 것은 매우 중요하다. 작은 기하학적 오류가 전체적인 위상(global topology)을 변화시킬 수 있으며, 이는 결과적으로 임상 의사결정 과정에서 기능적인 오류를 야기할 가능성이 크기 때문이다. 

기존의 딥러닝 기반 분할 네트워크는 주로 픽셀 단위의 분류(pixel-wise classification) 문제로 접근하며, 세그멘테이션 마스크만을 유일한 레이블로 사용한다. 그러나 이러한 방식은 픽셀 간의 관계나 기하학적 특성을 직접적으로 활용하지 못하므로, 공간적 일관성(spatial coherence)이 낮아지고 특히 노이즈가 많은 의료 데이터에서 위상적 결함이 발생하는 한계가 있다.

본 논문의 목표는 픽셀 연결성(pixel connectivity) 개념을 도입하여 픽셀 간의 관계를 모델링함으로써 해부학적으로 일관된 분할 결과를 생성하는 네트워크를 설계하는 것이다. 특히, 기존의 연결성 모델링 연구들이 잠재 공간(latent space) 내의 풍부한 채널별 방향성 정보(directional information)를 무시했다는 점에 착안하여, 이를 효과적으로 분리하고 활용하는 것을 핵심 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 잠재 공간 내에서 **방향성 서브스페이스(directional sub-space)를 공유 잠재 공간으로부터 분리(disentanglement)**하여 특징 표현(feature representation) 능력을 향상시키는 것이다.

이를 위해 저자들은 다음과 같은 설계를 제안한다.
1. **SDE(Sub-path Direction Excitation) 모듈**: 채널별 슬라이싱(channel-wise slicing)을 통해 범주적 특징(categorical features)과 방향성 특징을 분리하고, 각 서브패스에서 방향성 정보를 강화한다.
2. **IFD(Interactive Feature-space Decoder)**: 공간 흐름(space flow)과 특징 흐름(feature flow)이라는 두 개의 하향식(top-down) 상호작용 흐름을 통해 방향성 정보를 특징 맵에 효과적으로 융합한다.
3. **SDL(Size Density Loss)**: 의료 데이터셋에서 흔히 발생하는 레이블 크기의 불균형 문제를 해결하기 위해 레이블 크기 분포 기반의 가중치 스키마를 적용한 손실 함수를 제안한다.

## 📎 Related Works

### Deep connectivity modeling
디지털 위상학에서 픽셀 연결성은 인접한 픽셀들이 어떻게 관계를 맺고 있는지를 설명한다. 최근 딥러닝 분야에서는 세그멘테이션 마스크의 위상적 확장판인 연결성 마스크(connectivity mask)를 레이블로 사용하여 픽셀 간의 관계를 학습하는 시도들이 있었다. 하지만 이러한 기존 방식들은 범주적 정보와 방향성 정보가 하나의 네트워크 경로에서 동시에 학습되어 잠재 공간이 강하게 결합(coupled)되고 중복성이 발생하는 문제가 있었다.

### Interpretation of latent space
잠재 공간의 해석과 분리는 비지도 학습, 멀티모달 융합, 생성 모델 등에서 중요하게 다루어져 왔다. 본 논문은 연결성 마스크의 고유한 특성을 활용하여, 복잡한 VAE(Variational Autoencoder) 대신 단순하면서도 효과적인 서브패스 슬라이싱 방식을 통해 방향성 서브스페이스를 분리한다.

### Self-attention mechanism
Self-attention은 특징 맵 내의 픽셀 간 의존성을 캡처하는 데 널리 사용된다. 본 논문은 이러한 메커니즘을 SDE와 IFD 모듈에 통합하여 방향성 임베딩과 메인 특징 맵 사이의 관계를 모델링한다.

## 🛠️ Methodology

### 전체 시스템 구조
DconnNet은 사전 학습된 ResNet 인코더, SDE 모듈, 그리고 IFD 디코더로 구성된다. 전체적인 흐름은 인코더에서 추출된 특징을 SDE를 통해 방향성 특징으로 분리 및 강화하고, 이를 IFD를 통해 다시 세밀하게 융합하여 최종 연결성 출력을 생성하는 구조이다.

### Sub-path Direction Excitation (SDE) 모듈
SDE는 공유 잠재 공간에서 방향성 정보를 분리해내는 역할을 하며 세 단계로 진행된다.

1. **방향성 사전 정보 추출 (Directional Prior Extraction)**: 
   인코더의 마지막 출력 $e_N$을 업샘플링하여 예비 출력 $\hat{y}_{conn}$을 얻는다. 이를 글로벌 평균 풀링(GAP)하고 $1\times 1$ 컨볼루션을 통해 벡터화한 뒤, 시그모이드 함수 $\sigma$를 적용하여 방향성 사전 정보 $\alpha_{prior}$를 생성한다.
   $$\alpha_{prior} = \sigma(W_s e_{prior})$$
   여기서 $e_{prior}$는 $\text{ReLU}(W_k \text{GAP}(\hat{y}_{conn}))$로 계산된다.

2. **채널별 슬라이싱 (Channel-wise Slicing)**: 
   잠재 특징 $e_N$과 방향성 사전 정보 $\alpha_{prior}$를 채널 방향으로 8등분하여 $e_{N,i}$와 $\alpha_{prior,i}$로 나눈다. 이는 범주적 특징(중복성이 높음)과 방향성 특징(변별력이 높음)의 정보 밀도 차이를 이용하여 각 서브패스가 서로 다른 특징에 집중하게 만들기 위함이다.

3. **서브패스 흥분 (Sub-path Excitation, SPE)**: 
   각 슬라이스 쌍에 대해 공간 및 채널 어텐션을 적용하고, 방향성 사전 정보와 요소별 곱셈을 수행하여 특정 방향의 특징을 강조하거나 억제한다.
   $$e'_{i} = W_{1i}(\alpha_{prior,i} \cdot e'_{i,att}) + e_{N,i}$$
   최종적으로 8개의 서브패스 출력을 다시 쌓아(stack) 새로운 특징 맵 $e'$를 생성한다.

### Interactive Feature-space Decoder (IFD)
IFD는 두 가지 흐름의 상호작용을 통해 정보를 융합한다.

- **Space Flow**: 특징 맵 $f_i$에서 방향성 임베딩 $n_i = \text{GAP}(f_i)$를 추출하고, 이를 메인 특징 맵 $d_i$와 공유 매니폴드에 투영하여 유사도를 계산한다. 이 유사도 맵 $\alpha_{cat-dir} = \sigma(d'_{i} \cdot n'_{i})$를 사용하여 메인 특징 맵의 방향성 정보를 강화한다.
- **Feature Flow**: 단순한 컨볼루션 레이어와 업샘플링을 통해 메인 특징 맵 $d_i$를 상위 해상도로 전달하며, Space Flow에서 강화된 정보를 수신하여 다시 정교화한다.

### 손실 함수 및 최종 출력
최종 출력은 Bilateral Voting(BV)과 Region-guided Channel Aggregation(RCA)을 통해 세그멘테이션 맵으로 변환된다. 전체 손실 함수는 다음과 같다.
$$\mathcal{L}_{total} = \mathcal{L}(main) + 0.3 * \mathcal{L}(prior)$$

특히, 데이터 불균형을 해결하기 위해 **Size Density Loss (SDL)**를 제안한다. 이는 학습 전 레이블 크기(양성 픽셀 수)의 확률 밀도 함수 $P_{size}(k)$를 계산하고, 이에 반비례하는 가중치 $w(k)$를 Dice Loss에 적용하는 방식이다.
$$w(k) = \begin{cases} 1, & k=0 \\ -\log P_{size}(k), & k \neq 0 \end{cases}$$

## 📊 Results

### 실험 설정
- **데이터셋**: Retouch (망막 유체), ISIC2018 (피부 병변), CHASEDB1 (혈관)
- **평가 지표**: $\text{Dice}$, $\text{IOU}$, $\text{Accuracy}$, $\text{Precision}$, $\text{clDice}$, Betti numbers ($\beta_0, \beta_1$)

### 정량적 결과
- **Retouch**: SOTA 모델들(nnU-Net, CPFNet 등)보다 높은 $\text{Dice}$ 및 $\text{BACC}$를 기록하였다. 특히 작은 유체 영역에서의 예측 일관성이 크게 향상되었다.
- **ISIC2018**: $\text{Dice}$ 90.43%를 달성하며 비교 대상인 Ms RED 등의 모델을 앞질렀다.
- **CHASEDB1**: 위상 보존 능력을 평가하는 $\text{clDice}$와 Betti number에서 가장 우수한 성적을 거두어, 혈관의 연결성을 매우 잘 유지함을 증명하였다.

### 정성적 결과
시각화 결과, DconnNet은 다른 모델들이 놓치기 쉬운 아주 작은 영역(tiny SRF)을 정확히 찾아냈으며, 혈관 분할에서 나타나는 단절 현상이나 잘못된 연결(topological errors) 없이 매끄럽고 연결된 결과를 생성하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 단순한 픽셀 분류를 넘어 픽셀 간의 '연결성'과 '방향성'이라는 위상적 특성을 네트워크 구조에 직접 녹여냈다는 점에서 큰 강점이 있다. 특히 T-SNE 분석을 통해 SDE 모듈 적용 전후의 잠재 공간이 중심 집중형에서 양극화된 분포로 변화함을 보여줌으로써, 방향성 서브스페이스의 분리가 실제로 이루어졌음을 입증하였다.

### 한계 및 논의
실험 결과는 매우 우수하나, CHASEDB1 데이터셋의 경우 샘플 수가 매우 제한적이어서 SDL 손실 함수를 적용하지 못했다는 점이 언급되었다. 또한, 본 연구는 2D 이미지 기반의 연결성 모델링에 집중하고 있으나, 저자들은 향후 3D 의료 영상 분할로의 확장이 가능하며 파라미터 증가량이 적어 효율적일 것이라고 전망하고 있다.

## 📌 TL;DR

DconnNet은 의료 영상 분할에서 해부학적 일관성을 확보하기 위해 **잠재 공간 내의 방향성 특징을 분리(disentanglement)하고 강화**하는 네트워크이다. SDE 모듈을 통한 특징 분리와 IFD를 통한 인터랙티브 융합, 그리고 데이터 불균형을 해결하는 SDL 손실 함수를 통해 SOTA 성능을 달성하였다. 이 연구는 특히 혈관이나 신경과 같이 위상적 연결성이 중요한 의료 영상 분석 분야에서 매우 중요한 역할을 할 것으로 기대된다.