# SAMSNeRF: Segment Anything Model (SAM) Guides Dynamic Surgical Scene Reconstruction by Neural Radiance Field (NeRF)

Ange Lou, Yamin Li, Xing Yao, Yike Zhang, Jack Noble (2023)

## 🧩 Problem to Solve

본 논문은 수술 비디오로부터 수술 장면의 정밀한 3D 재구성(Reconstruction)을 수행하는 것을 목표로 한다. 수술 장면의 3D 재구성은 수술 중 내비게이션, 이미지 유도 로봇 수술 자동화, 증강 현실(AR) 및 수술 교육과 같은 다양한 임상 응용 분야에서 필수적인 선행 조건이다.

기존의 접근 방식들은 주로 Depth Estimation(깊이 추정)에 의존하였으나, 다음과 같은 한계가 존재한다:

1. **동적 객체 처리의 어려움**: 기존 방법들은 정적인 장면에서는 효과적이지만, 수술 도구와 같이 움직이는 객체가 포함된 동적 장면에서는 신뢰도가 급격히 떨어진다.
2. **텍스처 정보 손실**: 희소한 Warp Field 상에서 장면을 재구성함으로써 픽셀 간의 세밀한 텍스처 정보를 소실하는 문제가 발생한다.

따라서 본 연구는 수술 도구가 포함된 동적 수술 장면을 고충실도(High-fidelity)로 재구성하고, 모든 프레임에서 수술 도구의 정확한 3D 위치를 예측하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Segment Anything Model (SAM)**의 강력한 제로샷(Zero-shot) 세그멘테이션 능력을 **Neural Radiance Field (NeRF)** 기반의 동적 장면 재구성에 결합하는 것이다.

- **SAM을 통한 정밀한 마스크 생성**: 수동으로 지정한 Bounding Box를 프롬프트로 사용하여 수술 도구의 정밀한 세그멘테이션 마스크를 생성한다.
- **SAM 가이드 기반 Depth Refinement**: SAM이 제공하는 정확한 마스크를 활용하여, 배경 조직과 전경의 수술 도구 각각에 대해 깊이 지도를 정밀하게 보정(Refinement)함으로써 NeRF 학습의 불확실성을 제거한다.
- **동적 수술 장면의 3D 재구성**: 수술 도구가 포함된 동적 장면을 성공적으로 재구성한 최초의 방법론임을 주장한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 차별점을 제시한다:

- **Depth Estimation 기반 재구성**: 기존의 깊이 추정 방식은 정적 장면에서는 유효하지만, 움직이는 수술 도구가 있는 환경에서는 부적합하며 텍스처 손실이 발생한다.
- **Neural Radiance Fields (NeRF)**: NeRF는 정적 장면의 고충실도 뷰 합성(View Synthesis)에 탁월하며, D-NeRF 등 동적 장면으로 확장된 변형 모델들이 등장하였다.
- **EndoNeRF**: 수술 장면의 변형 가능한 조직을 재구성하기 위해 제안된 모델이나, 본 논문은 여기서 더 나아가 '움직이는 수술 도구'라는 불확실성을 해결하기 위해 SAM을 결합하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인

SAMSNeRF는 입력 비디오 프레임 $\{I_i\}$, SAM을 통해 생성된 수술 도구 마스크 $\{M_i\}$, 그리고 사전 학습된 네트워크로 얻은 coarse depth maps $\{D_i\}$를 입력으로 받는다. 시스템은 크게 Canonical Neural Radiance Field와 Time-dependent Neural Displacement Field로 구성된다.

### 2. 장면 표현 및 렌더링

- **Canonical Field**: 8개 층의 MLP $\mathcal{F}_\Theta(x, d)$를 통해 3D 좌표 $x \in \mathbb{R}^3$와 시점 방향 $d \in \mathbb{R}^3$를 RGB 색상 $c(x, d) \in \mathbb{R}^3$와 공간 점유율(Space Occupancy) $\sigma(x) \in \mathbb{R}$로 매핑한다.
- **Displacement Field**: 또 다른 8개 층의 MLP $\mathcal{G}_\Phi(x, t)$를 통해 시간 $t$에서의 점 $x$와 Canonical Field 상의 대응점 사이의 변위(Displacement)를 매핑한다.
- **결합 모델**: 임의의 시간 $t$에서 점 $x$의 색상과 점유율은 $\mathcal{F}_\Theta(x + \mathcal{G}_\Phi(x, t), d)$를 통해 얻는다. 고주파 정보 캡처를 위해 Position Encoding $\gamma(\cdot)$이 사용된다.

### 3. Volume Rendering 및 손실 함수

카메라 광선 $r(s) = o + sd$를 따라 샘플링된 점들에 대해, 렌더링된 색상 $\hat{C}$와 깊이 $\hat{D}$는 다음과 같이 계산된다:
$$\hat{C}(r(s)) = \sum_{j=1}^{m-1} c(x_j, d) \alpha_j, \quad \hat{D}(r(s)) = \sum_{j=1}^{m-1} s_j \alpha_j$$
여기서 $\alpha_j$는 다음과 같이 정의된다:
$$\alpha_j = (1 - \exp(-\sigma(x_j)\Delta s_j)) \exp(-\sum_{k=1}^{j-1} \sigma(x_k)\Delta s_k)$$

학습을 위한 손실 함수 $\mathcal{L}$은 렌더링된 결과와 실제 값 사이의 오차로 정의된다:
$$\mathcal{L}(r(s)) = \|\hat{C}(r(s)) - C[u, v]\|^2_2 + |\hat{D}(r(s)) - D[u, v]|$$

### 4. SAM 기반 Depth Refinement

정반사(Specular reflection)나 흐릿한 픽셀로 인해 발생하는 Depth Map의 오류를 해결하기 위해 다음과 같은 정밀화 과정을 거친다.

- **배경 정밀화 (Background Refinement)**: 배경의 잔차 맵(Residual map)을 $\epsilon_i^B = |\hat{D}_i^K - D_i| \odot (1 - M_i)$로 정의한다. 여기서 $\hat{D}_i^K$는 $K$번의 반복 학습 후 예측된 깊이다. 잔차가 큰 상위 $\alpha$-분위수(quantile) 픽셀들을 부드러운 배경 깊이 값으로 대체한다.
- **전경 정밀화 (Foreground Refinement)**: 수술 도구 영역에 대해 $\epsilon_i^F = |\hat{D}_i^K - D_i| \odot M_i$를 계산하고, 동일하게 $\alpha$-분위수의 픽셀들을 부드러운 전경 깊이 값으로 대체하여 보정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: EndoNeRF 데이터셋 (2개 케이스, 각각 63프레임과 156프레임으로 구성).
- **지표**: PSNR, SSIM, LPIPS를 사용하여 정량적으로 평가한다.
- **구현 세부사항**: Depth Refinement 반복 횟수 $K=4000$, RTX A5000 GPU 1대당 케이스별 약 12시간 학습. Coarse depth map 생성을 위해 STTR-light를 사용하였다.

### 2. 정량적 결과

실험 결과, 단순한 NeRF 구조보다 foreground/background refinement를 적용했을 때 성능이 비약적으로 향상되었으며, 특히 SAM을 사용했을 때 가장 높은 성능을 보였다.

| 방법론 | PSNR $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ |
| :--- | :---: | :---: | :---: |
| EndoNeRF | 21.435 | 0.720 | 0.287 |
| Ours w/ CNN | 32.127 | 0.904 | 0.112 |
| Ours w/ GT | 34.425 | 0.918 | 0.100 |
| **Ours w/ SAM (SAMSNeRF)** | **34.537** | **0.921** | **0.095** |

### 3. 정성적 결과

- **포인트 클라우드 시각화**: 단순 Depth Map 기반 재구성보다 SAMSNeRF가 더 밀도 높은 포인트 클라우드를 생성하며 풍부한 텍스처 정보를 유지함을 확인하였다.
- **고충실도 재구성**: 배경 조직뿐만 아니라 수술 도구의 복잡한 세부 사항까지 성공적으로 재구성하였다.

## 🧠 Insights & Discussion

### 강점

- **SAM의 제로샷 활용**: 별도의 파인튜닝 없이 Bounding Box 프롬프트만으로 수술 도구를 정밀하게 분리해낼 수 있었으며, 이것이 Depth Refinement의 성능 향상으로 직결되었다.
- **동적 장면 해결**: 기존 NeRF 기반 수술 재구성 모델들이 간과했던 '움직이는 도구' 문제를 효율적으로 해결하여 실용적인 3D 정보를 제공할 수 있게 되었다.

### 한계 및 미해결 과제

- **학습 시간**: 단일 케이스를 학습시키는 데 12시간이 소요되어 실시간 응용에는 한계가 있다.
- **사전 깊이 지도 의존성**: 학습 전 단계에서 사전 학습된 Depth Map이 필요하다는 점이 제약 사항으로 작용한다.

### 비판적 해석

본 논문은 SAM의 강력한 세그멘테이션 성능을 NeRF의 최적화 과정에 적절히 편입시켜 수술 장면이라는 특수 도메인에서의 성능을 끌어올렸다. 특히 Depth Map의 노이즈를 처리하기 위한 $\alpha$-quantile 기반의 정밀화 전략이 PSNR 수치를 크게 높이는 결정적 역할을 한 것으로 보인다. 다만, 정량적 표에서 SAM의 결과가 Ground Truth(GT)보다 높게 나타난 점은 SAM의 마스크가 매우 정밀하거나 혹은 평가 데이터셋의 특성상 발생한 결과로 추측되며, 이에 대한 구체적인 분석이 부족하다.

## 📌 TL;DR

본 논문은 수술 비디오에서 움직이는 도구를 포함한 동적 3D 장면을 재구성하기 위해 **SAM(Segment Anything Model)**과 **NeRF**를 결합한 **SAMSNeRF**를 제안한다. SAM으로 생성한 정밀한 도구 마스크를 이용해 깊이 지도를 보정함으로써, 기존 방식보다 훨씬 정밀한 고충실도 3D 재구성을 달성하였다. 이는 향후 수술 내비게이션 및 로봇 수술 자동화에서 도구의 정확한 3D 위치 정보를 제공하는 데 중요한 기반 기술이 될 것으로 기대된다.
