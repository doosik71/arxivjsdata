# ZigMa: A DiT-style Zigzag Mamba Diffusion Model

Vincent Tao Hu, Stefan Andreas Baumann, Ming Gui, Olga Grebenkova, Pingchuan Ma, Johannes Schusterbauer, and Björn Ommer (2024)

## 🧩 Problem to Solve

본 논문은 이미지 및 비디오 생성 분야에서 널리 사용되는 Diffusion Model의 확장성(Scalability)과 계산 복잡도 문제를 해결하고자 한다. 특히, 최근 Transformer 기반의 구조(예: DiT)가 높은 확장성을 보여주며 주류가 되었으나, Attention 메커니즘 특유의 시퀀스 길이에 대한 이차 복잡도($\text{quadratic complexity}$)는 고해상도 데이터 처리 시 심각한 병목 현상을 야기한다.

이러한 문제를 해결하기 위해 선형 복잡도를 가진 State-Space Model(SSM)의 최신 아키텍처인 Mamba가 대안으로 제시되고 있다. 그러나 Mamba는 기본적으로 1D 시퀀스 모델링을 위해 설계되었으며, 이를 2D 이미지나 3D 비디오에 적용하기 위해서는 데이터를 평탄화(Flattening)하는 과정이 필요하다. 기존의 Mamba 기반 비전 모델들은 단순히 행-열 우선 순서(row-and-column-major order)로 토큰을 배치함으로써 **공간적 연속성(Spatial Continuity)**을 간과하는 경향이 있으며, 이를 보완하기 위해 여러 방향으로 스캔하는 방식은 추가적인 파라미터와 메모리 부담을 초래한다는 문제가 있다.

따라서 본 연구의 목표는 추가적인 파라미터 부담 없이 이미지의 공간적 연속성을 보존하는 효율적인 스캔 방식을 제안하고, 이를 Diffusion Model의 백본으로 통합하여 고해상도 이미지 및 비디오 생성에서의 효율성과 성능을 입증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 스캔 경로를 지그재그(Zigzag) 형태로 구성하여 시각 데이터의 공간적 연속성을 최대한 활용하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Zigzag Mamba (ZigMa) 제안**: 추가 파라미터 없이 레이어별로 서로 다른 지그재그 스캔 경로를 적용하는 Heterogeneous Layerwise Scan 패러다임을 도입하였다. 이를 통해 2D 이미지의 공간적 인덕티브 바이어스(Inductive Bias)를 효율적으로 통합하였다.
2. **3D 비디오 확장**: 공간(Spatial)과 시간(Temporal) 차원을 분리하여 모델링하는 Factorized 3D Zigzag 구조를 제안하여, 3D 데이터의 최적화 문제를 해결하였다.
3. **Stochastic Interpolant 프레임워크 통합**: 제안한 Zigzag Mamba를 Stochastic Interpolant 프레임워크에 결합하여 $1024 \times 1024$ 고해상도 이미지 및 비디오 생성에서의 확장성을 검증하였다.
4. **효율성 입증**: Transformer 기반 모델(DiT, U-ViT) 대비 메모리 사용량과 추론 속도(FPS) 면에서 우위를 점하면서도, 기존 Mamba 기반 베이스라인보다 뛰어난 생성 성능을 보임을 확인하였다.

## 📎 Related Works

### Mamba 및 SSM

Mamba는 선택적 상태 공간(Selective State Spaces)을 통해 긴 시퀀스를 선형 시간 복잡도로 처리할 수 있는 능력을 갖추고 있다. VisionMamba, S4ND, Mamba-ND 등이 이를 비전 분야에 적용하려 시도하였다. VisionMamba는 양방향 SSM을 사용하지만 계산 비용이 높으며, Mamba-ND는 단일 블록 내에서 다양한 스캔 방향을 고려하여 파라미터 및 메모리 부담이 증가한다. 본 논문은 이러한 복잡도를 단일 블록이 아닌 전체 네트워크의 레이어별로 분산 배치함으로써 효율성을 극대화한다.

### Diffusion Model 백본

기존 Diffusion Model은 주로 UNet 기반 구조를 사용했으나, 최근에는 확장성이 뛰어난 ViT(Vision Transformer) 기반의 DiT 구조가 주목받고 있다. 하지만 ViT의 이차 복잡도는 여전히 한계점으로 남는다. DiffSSM과 같은 연구가 SSM을 Diffusion 백본으로 시도했으나, 본 논문은 더 복잡한 고해상도 데이터와 텍스트 조건부 생성으로 범위를 확장하고 구체적인 스캔 경로 최적화에 집중했다는 점에서 차별화된다.

### SDE 및 ODE 기반 생성 모델

SMLD, DDPM 등은 확률 미분 방정식(SDE) 프레임워크 내에서 작동한다. 최근에는 ODE 샘플러를 통해 샘플링 비용을 줄이려는 시도가 많으며, Flow Matching이나 Rectified Flow 등이 이에 해당한다. 본 논문은 이러한 다양한 생성 모델을 통합하는 Stochastic Interpolant 프레임워크를 채택하여 일반성을 확보하였다.

## 🛠️ Methodology

### 전체 시스템 구조

ZigMa는 DiT의 구조를 계승하여 Adaptive Layer Norm(AdaLN)을 사용하며, 핵심 추론 모듈로 **Single-scan Mamba Block**을 사용한다. 전체 네트워크는 $L$개의 레이어로 구성되며, 각 레이어는 서로 다른 지그재그 스캔 경로를 통해 데이터를 처리한다.

### Zigzag Scanning 메커니즘

이미지 패치 토큰들이 Mamba 블록에 입력되기 전, 토큰의 순서를 재배치하는 $\Omega$ 연산을 수행한다. 과정은 다음과 같다.

1. **재배치(Arrange)**: 입력 특징 $z_i$를 $\Omega_i$ 순서로 재배치하여 $z_{\Omega_i}$를 생성한다.
   $$z_{\Omega_i} = \text{arrange}(z_i, \Omega_i)$$
2. **스캔(Scan)**: 재배치된 시퀀스를 Mamba의 Forward Scan 블록으로 처리한다.
   $$\bar{z}_{\Omega_i} = \text{scan}(z_{\Omega_i})$$
3. **역재배치(Reverse Arrange)**: 처리된 결과를 다시 원래의 이미지 토큰 순서로 복구한다.
   $$z_{i+1} = \text{arrange}(\bar{z}_{\Omega_i}, \bar{\Omega}_i)$$

여기서 $\Omega_i$는 공간적 연속성을 보존하는 8가지의 서로 다른 공간 채움(Space-filling) 경로 $S_j (j \in [0, 7])$ 중 하나로 선택되며, 레이어 인덱스 $i$에 대해 $\Omega_i = S_{\{i \pmod 8\}}$로 설정하여 레이어마다 다른 경로를 갖게 한다.

### 텍스트 조건부 생성 (Text-Conditioning)

Mamba는 Attention 메커니즘이 없으므로 텍스트 조건 주입이 어렵다. 이를 해결하기 위해 Mamba 블록 상단에 **Cross-Attention 블록**을 추가하였다. 텍스트 프롬프트와 타임스텝 정보는 MLP를 통해 임베딩되어 Mamba 스캔과 Cross-Attention 모듈을 각각 변조(Modulate)한다.

### 3D 비디오 확장: Factorized 3D Zigzag

3D 데이터의 경우 단순한 3D 지그재그 스캔은 최적화가 어렵다는 점을 발견하였다. 따라서 공간적 상관관계와 시간적 상관관계를 분리하여 처리하는 **Factorized 3D Zigzag** 방식을 제안한다.

- **Spatial-zigzag Mamba**: 2D 이미지에서 제안한 지그재그 스캔을 각 프레임에 적용한다.
- **Temporal-zigzag Mamba**: 시간 축을 따라 1D 스윕(Sweep) 스캔을 적용한다.
이 두 블록을 "sstt" 또는 "ststst"와 같은 순서로 교차 배치하여 효율적으로 모델링한다.

### 훈련 프레임워크: Stochastic Interpolant

모델은 Stochastic Interpolant 프레임워크 하에서 학습된다. 데이터 분포 $p_0(x)$와 노이즈 분포 $p_T(x)$를 연결하는 경로를 학습하며, 속도 필드 $v(x, t)$ 또는 스코어 필드 $s(x, t)$를 추정한다.
본 논문은 선형 경로($\alpha_t = 1-t, \sigma_t = t$)를 채택하였으며, 손실 함수는 다음과 같이 정의된다.
$$\mathcal{L}_v(\theta) = \int_{0}^{T} \mathbb{E}[\|v_\theta(x_t, t) - \dot{\alpha}_t x^* - \dot{\sigma}_t \varepsilon\|^2] dt$$
여기서 $\theta$는 Zigzag Mamba 네트워크를 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋**: FacesHQ ($1024 \times 1024$), MS-COCO ($256 \times 256$), MultiModal-CelebA, UCF101 (비디오).
- **지표**: FID, FDD, KID (이미지), Frame-FID, FVD (비디오).
- **베이스라인**: VisionMamba (Bidirectional Mamba), DiT, U-ViT.

### 주요 결과

1. **고해상도 이미지 생성**: $1024 \times 1024$ FacesHQ 데이터셋에서 ZigMa는 Bidirectional Mamba보다 우수한 FID(37.8 vs 51.1)를 기록하였다. 특히 배치 사이즈를 키웠을 때 성능 향상이 뚜렷했다.
2. **스캔 경로의 영향**: Ablation study 결과, 단순 Sweep 스캔보다 Zigzag-1이, Zigzag-1보다 8가지 경로를 교차 사용하는 Zigzag-8이 훨씬 뛰어난 성능을 보였다. 이는 고해상도일수록 효과가 더 컸다.
3. **효율성 분석**:
    - **메모리 및 속도**: 패치 수가 증가함에 따라 DiT나 U-ViT는 메모리 사용량이 급격히 증가하고 FPS가 하락하지만, ZigMa는 선형적인 복잡도 덕분에 훨씬 적은 메모리로 더 빠른 속도를 유지하였다.
    - **파라미터 효율성**: Order Receptive Field(사용된 지그재그 경로의 수)를 8까지 늘려도 추가 파라미터나 메모리 증가 없이 성능만 향상되었다.
4. **비디오 생성**: UCF101 데이터셋에서 Factorized 3D Zigzag 방식이 Bidirectional Mamba 기반 방식보다 낮은 Frame-FID 및 FVD를 기록하여 우수성을 입증하였다.

## 🧠 Insights & Discussion

### 공간적 연속성의 중요성

본 논문은 Mamba를 비전 데이터에 적용할 때 단순한 평탄화가 아닌 **공간적 연속성(Spatial Continuity)**을 확보하는 것이 성능의 핵심임을 밝혔다. 실험을 통해 패치 그룹 크기를 키워 연속성을 높였을 때 FID가 개선됨을 확인하였으며, 이는 Mamba가 시퀀스 내 인접 토큰 간의 관계를 학습하는 능력이 중요함을 시사한다.

### 힐버트 곡선(Hilbert Curve)과의 비교

더 정교한 공간 채움 곡선인 힐버트 곡선을 실험하였으나, 결과적으로 Zigzag 스캔보다 성능이 낮았다. 저자들은 힐버트 곡선의 지나치게 복잡한 구조가 오히려 SSM의 최적화를 방해한다고 분석하며, 생성 작업에서는 국소성(Locality)보다 단순하고 명확한 구조적 패턴이 더 중요할 수 있다는 가설을 제시하였다.

### 한계점 및 비판적 해석

- **경로 설계의 휴리스틱**: 8가지 지그재그 경로를 휴리스틱하게 설계하였으므로, 패치 사이즈에 최적화된 자동화된 경로 탐색 방법이 부재하다는 점이 한계로 지적된다.
- **훈련 시간의 제약**: GPU 자원 제한으로 인해 충분히 긴 시간 동안 학습시키지 못해, 더 큰 규모의 학습 시 잠재력이 더 클 수 있음을 언급하고 있다.

## 📌 TL;DR

본 논문은 Mamba의 선형 복잡도 장점을 유지하면서 2D/3D 데이터의 공간적 연속성을 보존하는 **Zigzag Mamba(ZigMa)** 구조를 제안하였다. 레이어별로 다른 지그재그 스캔 경로를 적용하는 전략을 통해 추가 파라미터 없이 인덕티브 바이어스를 극대화하였으며, 이를 통해 $1024 \times 1024$ 고해상도 이미지 생성에서 Transformer 기반 모델보다 효율적이고 기존 Mamba 기반 모델보다 뛰어난 성능을 달성하였다. 이 연구는 향후 초고해상도 시각 콘텐츠 생성 및 효율적인 비디오 모델링을 위한 중요한 설계 방향을 제시한다.
